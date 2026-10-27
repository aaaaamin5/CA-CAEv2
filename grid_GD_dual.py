
# conda activate r423
# python /public/home/tanjingbo/CA-CAE-main/grid_GD_dual.py


# ==============================================================================
# 项目名称: 多视图癌症亚型识别 (Graph Denoising + CA-CAE + Concat Baseline)
# 核心逻辑: 单组学拓扑图平滑降噪 -> CA-CAE 提取 -> 直接级联融合 -> 三维网格搜索
# ==============================================================================
import os

# 配置 R 语言环境路径
# os.environ['R_HOME'] = '/public/home/tanjingbo/miniconda3/envs/r423/lib/R'
# os.environ['PATH'] = '/public/home/tanjingbo/miniconda3/envs/r423/bin:' + os.environ.get('PATH', '')
# # 额外加上LD_LIBRARY_PATH，rpy2加载libR.so需要
# os.environ['LD_LIBRARY_PATH'] = '/public/home/tanjingbo/miniconda3/envs/r423/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro

import datetime
import csv
import warnings
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ================= 1. 🌟 强制全局随机种子固定 =================
def setup_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    ro.r(f'set.seed({seed})')
    print(f"🔒 全局随机种子已固定为: {seed} (覆盖 Python, NumPy, TensorFlow, R)")

setup_seed(42)

# ================= 2. 🌟 必不可少的 Pandas 降级补丁 =================
pd.Series.iteritems = pd.Series.items  
pd.DataFrame.iteritems = pd.DataFrame.items
original_describe = pd.DataFrame.describe
def patched_describe(self, *args, **kwargs):
    kwargs.pop('datetime_is_numeric', None)
    return original_describe(self, *args, **kwargs)
pd.DataFrame.describe = patched_describe




# ==============================================================================
# Branch 2: 基于 SSLNMVC 的共识特征提取网络 (U-Projection)
# 源码出处: network.py 中的 map_layer, fusion 和 forwardZ
# ==============================================================================
class SSLNMVC_CommonBranch(tf.keras.Model):
    def __init__(self, hidden_dim, mlp_layers=1, use_bn=False):
        super(SSLNMVC_CommonBranch, self).__init__()
        self.hidden_dim = hidden_dim
        latent_dim = hidden_dim // 2  # 对应源码: latent_dim = args.hidden_dim // 2

        # 1. 映射层: 对应源码 self.map_layer[cite: 15]
        self.map_layer = tf.keras.Sequential(name="map_layer")
        self.map_layer.add(tf.keras.layers.Dense(hidden_dim, use_bias=True))
        if use_bn:
            self.map_layer.add(tf.keras.layers.BatchNormalization())
        self.map_layer.add(tf.keras.layers.ReLU())

        # 2. U-Pro 融合层 (沙漏结构): 对应源码 self.fusion 与 _make_layers[cite: 15]
        self.fusion = tf.keras.Sequential(name="fusion")
        # 降维 (Downsample)
        self.fusion.add(tf.keras.layers.Dense(latent_dim, use_bias=False))
        if use_bn:
            self.fusion.add(tf.keras.layers.BatchNormalization())
        self.fusion.add(tf.keras.layers.ReLU())
        
        # 中间隐藏层
        for _ in range(mlp_layers):
            self.fusion.add(tf.keras.layers.Dense(latent_dim, use_bias=False))
            if use_bn:
                self.fusion.add(tf.keras.layers.BatchNormalization())
            self.fusion.add(tf.keras.layers.ReLU())
            
        # 升维 (Upsample)
        self.fusion.add(tf.keras.layers.Dense(hidden_dim, use_bias=False))

    def call(self, h_concat, training=False):
        """
        对应源码的 forwardZ 逻辑
        """
        # 1. 维度对齐映射
        z = self.map_layer(h_concat, training=training)
        res = z  # 保存残差
        
        # 2. 沙漏网络提纯
        z_fused = self.fusion(z, training=training)
        
        # 3. 残差相加，防止信息丢失
        commonZ = z_fused + res
        return commonZ

# ==============================================================================
# 分支关系函数：正交约束 (Orthogonality Constraint)
# 作用：强迫 Branch 2 学到的共识特征不重复提取 Branch 1 的私有特征，保证方差满秩
# ==============================================================================
def orthogonality_loss(z_common, z_specific_list):
    loss = 0.0
    z_common_norm = tf.math.l2_normalize(z_common, axis=0)
    for z_spec in z_specific_list:
        z_spec_norm = tf.math.l2_normalize(z_spec, axis=0)
        # 计算互相关矩阵并最小化 Frobenius 范数
        corr_matrix = tf.matmul(z_common_norm, z_spec_norm, transpose_a=True)
        loss += tf.reduce_mean(tf.square(corr_matrix))
    return loss

# ================= 3. 🌟 核心新增：单组学拓扑特征降噪 (Graph Smoothing) =================
def topological_denoising(df_features, k=5):
    """
    基于 k-NN 邻接矩阵的单组学拓扑特征降噪 (模拟 GCN 聚合)。
    :param df_features: 输入的组学 DataFrame (行: 样本, 列: 特征)
    :param k: 聚类邻居数
    :return: 降噪后的 DataFrame
    """
    # 构建无向 k-NN 邻接矩阵
    A = kneighbors_graph(df_features.values, n_neighbors=k, mode='connectivity', include_self=False)
    A = A.maximum(A.T)  # 强制对称
    
    # 增加自环
    I = sp.eye(A.shape[0])
    A_tilde = A + I
    
    # 度矩阵对称归一化 (D^{-1/2} A D^{-1/2})
    D_tilde = np.array(A_tilde.sum(axis=1)).flatten()
    D_inv_sqrt = np.power(D_tilde, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    D_inv_sqrt_mat = sp.diags(D_inv_sqrt)
    
    # 拉普拉斯平滑算子
    S = D_inv_sqrt_mat.dot(A_tilde).dot(D_inv_sqrt_mat)
    
    # 图信号平滑滤波
    X_smooth = S.dot(df_features.values)
    
    return pd.DataFrame(X_smooth, index=df_features.index, columns=df_features.columns)

# ================= 4. 全局配置与 R 语言初始化 =================
pandas2ri.activate()
print("📦 正在加载 R 语言严格评估模块...")
ro.r.source('/public/home/tanjingbo/CA-CAE-main/CACAE/strict_eval.R')
analyze_clusters_r = ro.globalenv['analyze_clusters']

# cancer_datasets = ['KIRC','BLCA','BRCA','PAAD','STAD','GBM','SKCM','UCEC','UVM','LUAD']
cancer_datasets = ['SARC']
# cancer_datasets = ['PAAD','GBM','UVM']
# cancer_datasets = ['ACC','CESC','CHOL','COAD','DLBC','KICH','LAML','LIHC','LUSC','MESO','OV','SARC']
# cancer_datasets = ['COAD','DLBC','KICH','LAML','LIHC','MESO']
# 搜索空间定义
p_thresh_values = [0.005]
# p_thresh_values = [0.005, 0.01, 0.15, 0.3]

lasso_alphas = [0.01]
# lasso_alphas = [0.01, 0.05, 0.1, 0.5, 1]

# k_values = [2,3,4,5]
# k_values = [6,7,8]
k_values = [5]



use_graph_denoise = True   # True: 使用平滑；False: 消融平滑
cluster_method = 'srsc'   # 可选: 'kmeans' 或 'srsc'


# 独立输出目录设计，避免与前序版本冲突
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
base_run_dir = f'/public/home/tanjingbo/CA-CAE-main/CACAE/GD_Dual_for_new/Run_{timestamp}_GraphDenoise_{cluster_method.upper()}'

os.makedirs(base_run_dir, exist_ok=True)

summary_csv = f'{base_run_dir}/PanCancer_SOTA_Summary.csv'
with open(summary_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    # 🚨 确保包含 Best_P_Thresh
    writer.writerow(['Cancer_Type', 'Best_P_Thresh', 'Best_Alpha', 'Best_K', 'Selected_Features', 'Sil_Score', 'Strict_neg_log_p', 'Strict_Enrichment'])

print(f"📂 本次图降噪实验所有结果将保存在独立文件夹: {base_run_dir}")

# ================= 5. 主循环 =================
for cancer_name in cancer_datasets:
    print("\n" + "🚀"*20)
    print(f"   正在全自动处理数据集: {cancer_name}")
    print("🚀"*20)

    cancer_out_dir = f'/public/home/tanjingbo/CA-CAE-main/CACAE/GD_Dual_for_new/{cancer_name}'
    os.makedirs(cancer_out_dir, exist_ok=True)

    try:
        # --- 5.1 数据加载 ---
        path2 = f"/public/home/tanjingbo/CA-CAE-main/CACAE/Data/fea/{cancer_name}/miRNA.fea"
        path3 = f"/public/home/tanjingbo/CA-CAE-main/CACAE/Data/fea/{cancer_name}/rna.fea"
        path4 = f"/public/home/tanjingbo/CA-CAE-main/CACAE/Data/fea/{cancer_name}/meth.fea"
        path5 = f"/public/home/tanjingbo/CA-CAE-main/CACAE/Data/clinic/{cancer_name}.clinic"
        path_cn = f"/public/home/tanjingbo/CA-CAE-main/CACAE/Data/fea/{cancer_name}/CN.fea" 

        miRNA = pd.read_table(path2, sep=',', index_col=0).T
        RNA = pd.read_table(path3, sep=',', index_col=0).T
        Meth = pd.read_table(path4, sep=',', index_col=0).T
        survive = pd.read_table(path5, sep=',', index_col=0)
        CN = pd.read_table(path_cn, sep=',', index_col=0).T 

        total_train_samples = RNA.shape[0]
        print(f"\n[*] 侦测到当前癌症 {cancer_name} 的组学样本总数为: {total_train_samples}")
        
        # --- 5.2 动态超参数分配 ---
        if total_train_samples < 150:
            dynamic_batch_size, dynamic_epochs = 8, 200
        elif total_train_samples < 400:
            dynamic_batch_size, dynamic_epochs = 16, 300
        elif total_train_samples < 800:
            dynamic_batch_size, dynamic_epochs = 32, 500
        else:
            dynamic_batch_size, dynamic_epochs = 64, 500

        dynamic_val_rate = 0.1 if total_train_samples < 100 else (0.15 if total_train_samples < 300 else 0.2)
        print(f"  -> 分配结果: Batch Size = {dynamic_batch_size} | Val Rate = {dynamic_val_rate} | Epochs = {dynamic_epochs}\n")

        from CACAE import Model, Process
        from CACAE.utils import ClusterProcessor
        from CACAE.Survive_select import survive_select

        # --- 5.3 基础特征筛选与数据插补 (Data Imputation) ---
        # 强制填充 NaN 为 0，切断缺失值向 K-NN 距离矩阵和深度网络的蔓延
        RNA = Process.DataProcessor(RNA).sort_corr(3200).fillna(0)
        miRNA = Process.DataProcessor(miRNA).sort_corr(300).fillna(0)
        Meth = Process.DataProcessor(Meth).sort_corr(3100).fillna(0)
        CN = Process.DataProcessor(CN).sort_corr(3100).fillna(0)
        
        # --- 5.4 强制归一化防御 ---
        print("\n⚙️ 正在执行全局 Min-Max 归一化，防止极值引发网络 NaN...")
        scaler = MinMaxScaler()
        RNA = pd.DataFrame(scaler.fit_transform(RNA), index=RNA.index, columns=RNA.columns)
        miRNA = pd.DataFrame(scaler.fit_transform(miRNA), index=miRNA.index, columns=miRNA.columns)
        Meth = pd.DataFrame(scaler.fit_transform(Meth), index=Meth.index, columns=Meth.columns)
        CN = pd.DataFrame(scaler.fit_transform(CN), index=CN.index, columns=CN.columns)

        # --- 5.5 单组学拓扑特征平滑 (Graph Denoising) ---
        if use_graph_denoise:
            print("\n⚙️ 正在执行单组学拓扑特征降噪 (Graph Smoothing)...")
            k_neighbors = 3 if total_train_samples < 200 else 5
            RNA = topological_denoising(RNA, k=k_neighbors)
            miRNA = topological_denoising(miRNA, k=k_neighbors)
            Meth = topological_denoising(Meth, k=k_neighbors)
            CN = topological_denoising(CN, k=k_neighbors)
            print(f"    [*] 拓扑降噪完成，邻居聚合参数 k={k_neighbors}。")
        else:
            print("\n⚙️ 当前为消融实验：跳过 Graph Smoothing，归一化后直接进入 CA-CAE。")


        # --- 5.6 初始化并训练 CA-CAE ---
        print("\n开始初始化并配置 CA-CAE 模型...")
        RNA_model = Model.CACAE(RNA.shape[1], epochs=dynamic_epochs, batch_size=dynamic_batch_size, validation_rate=dynamic_val_rate)
        miRNA_model = Model.CACAE(miRNA.shape[1], epochs=dynamic_epochs, batch_size=dynamic_batch_size, validation_rate=dynamic_val_rate)
        Meth_model = Model.CACAE(Meth.shape[1], epochs=dynamic_epochs, batch_size=dynamic_batch_size, validation_rate=dynamic_val_rate)
        CN_model = Model.CACAE(CN.shape[1], epochs=dynamic_epochs, batch_size=dynamic_batch_size, validation_rate=dynamic_val_rate) 
        
        print("\n开始训练网络 (耐心等待)...")
        RNA_model.fit(RNA)
        miRNA_model.fit(miRNA)
        Meth_model.fit(Meth)
        CN_model.fit(CN) 

        z_rna = RNA_model.extract_feature(RNA)
        z_mirna = miRNA_model.extract_feature(miRNA)
        z_meth = Meth_model.extract_feature(Meth)
        z_cn = CN_model.extract_feature(CN)


        # 转为 tf.Tensor 便于后续处理
        z_spec_list = [
            tf.cast(z_rna, tf.float32), 
            tf.cast(z_mirna, tf.float32), 
            tf.cast(z_meth, tf.float32), 
            tf.cast(z_cn, tf.float32)
        ]
        
        # 初步拼接用于输入第二分支 (对应源码 forwardZ 中的 torch.cat(h, dim=1)[cite: 15])
        h_concat = tf.concat(z_spec_list, axis=1)

        # ==========================================================
        # [核心调用点]: Branch 2 初始化与正交约束训练 (已修复 Batch 对齐 Bug)
        # ==========================================================
        print("\n开始训练 SSLNMVC 共识特征提取分支 (Branch 2)...")
        hidden_dim = h_concat.shape[1] // 2 
        common_branch = SSLNMVC_CommonBranch(hidden_dim=hidden_dim, mlp_layers=1, use_bn=False)
        common_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
        # [修复 1]: 增加 drop_remainder=True，丢弃尾部残缺 Batch，保护 L2 归一化与正交约束的稳定性
        dataset = tf.data.Dataset.from_tensor_slices(
            (h_concat, z_rna, z_mirna, z_meth, z_cn)
        ).batch(dynamic_batch_size, drop_remainder=True)
        
        for epoch in range(50):
            # 迭代时，直接解包出当前 batch 的所有特征
            for batch_h, b_rna, b_mirna, b_meth, b_cn in dataset:
                with tf.GradientTape() as tape:
                    # 1. 前向传播提取共识特征 (batch_h 来源于 for 循环的解包)
                    # commonZ = common_branch(batch_h)
                    commonZ = common_branch(batch_h, training=True)
                    
                    # 2. 组装当前严格对齐的 batch 私有特征列表 (需转为 float32 以匹配类型)
                    batch_z_spec_list = [
                        tf.cast(b_rna, tf.float32), 
                        tf.cast(b_mirna, tf.float32), 
                        tf.cast(b_meth, tf.float32), 
                        tf.cast(b_cn, tf.float32)
                    ]
                    
                    # 3. 计算正交约束损失
                    loss_ortho = orthogonality_loss(commonZ, batch_z_spec_list)
                    
                # 4. 梯度阻断：仅更新 Branch 2 的 U-Projection 网络权重
                grads = tape.gradient(loss_ortho, common_branch.trainable_variables)
                common_optimizer.apply_gradients(zip(grads, common_branch.trainable_variables))

        # ==========================================================
        # [末端融合]: 私有特征 + 共识特征
        # ==========================================================
        # 获取最终训练好的共识特征
        z_common_final = common_branch(h_concat).numpy()
        
        # 封装为 DataFrame
        RNA_feature = pd.DataFrame(z_rna, columns=[f"{cancer_name}_RNA_{i}" for i in range(z_rna.shape[1])])
        miRNA_feature = pd.DataFrame(z_mirna, columns=[f"{cancer_name}_miRNA_{i}" for i in range(z_mirna.shape[1])])
        Meth_feature = pd.DataFrame(z_meth, columns=[f"{cancer_name}_Meth_{i}" for i in range(z_meth.shape[1])])
        CN_feature = pd.DataFrame(z_cn, columns=[f"CN_{i}" for i in range(z_cn.shape[1])])
        Common_feature = pd.DataFrame(z_common_final, columns=[f"CommonZ_{i}" for i in range(z_common_final.shape[1])])

        # 两分支彻底融合 (Concat)，送入下游 Lasso 和 SRSC
        flatten = pd.concat([RNA_feature, miRNA_feature, Meth_feature, CN_feature, Common_feature], axis=1)
        flatten.index = RNA.index





        # --- 5.7 特征提取与融合 (Concat) ---
        # RNA_feature = pd.DataFrame(RNA_model.extract_feature(RNA), columns=[f"{cancer_name}_RNA_feat_{i}" for i in range(RNA_model.extract_feature(RNA).shape[1])])
        # miRNA_feature = pd.DataFrame(miRNA_model.extract_feature(miRNA), columns=[f"{cancer_name}_miRNA_feat_{i}" for i in range(miRNA_model.extract_feature(miRNA).shape[1])])
        # Meth_feature = pd.DataFrame(Meth_model.extract_feature(Meth), columns=[f"{cancer_name}_Meth_feat_{i}" for i in range(Meth_model.extract_feature(Meth).shape[1])])
        # CN_feature = pd.DataFrame(CN_model.extract_feature(CN), columns=[f"CN_feat_{i}" for i in range(CN_model.extract_feature(CN).shape[1])])

        # flatten = pd.concat([RNA_feature, miRNA_feature, Meth_feature, CN_feature], axis=1)

        # --- 5.8 临床 ID 对齐补丁 ---
        flatten.index = RNA.index 
        survive.index = survive.index.astype(str).str.upper().str.replace('.', '-').str[:12]
        flatten.index = flatten.index.astype(str).str.upper().str.replace('.', '-').str[:12]

        survive = survive[~survive.index.duplicated(keep='first')]
        flatten = flatten[~flatten.index.duplicated(keep='first')]

        common_samples = survive.index.intersection(flatten.index)
        survive = survive.loc[common_samples]
        flatten = flatten.loc[common_samples]
        
        # [修复 2]: 严禁对生存数据使用 fillna(0)
        survive['days'] = pd.to_numeric(survive['days'], errors='coerce')
        survive['status'] = pd.to_numeric(survive['status'], errors='coerce')
        
        # 必须剔除生存天数缺失或为 0 的无效样本 (Cox 模型物理禁忌)
        survive = survive.dropna(subset=['days', 'status'])
        survive = survive[survive['days'] > 0]
        
        # 仅对特征矩阵填 0
        flatten = flatten.fillna(0)
        
        # 重新对齐 (因为 survive 删除了无效行)
        common_samples_final = survive.index.intersection(flatten.index)
        survive = survive.loc[common_samples_final]
        flatten = flatten.loc[common_samples_final]
        print(f"[*] ✅ 对齐与清洗完成，共有 {len(common_samples_final)} 个有效生存样本进入筛选！")

        # ================= 6. 多维网格搜索 =================
        cancer_detail_csv = f'{cancer_out_dir}/{cancer_name}_GD_Dual_Uproject_New_{cluster_method.upper()}_GridSearch.csv'

        file_exists = os.path.isfile(cancer_detail_csv)
        
        with open(cancer_detail_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Timestamp', 'P_Thresh', 'Lasso_Alpha', 'K', 'Selected_Features', 'Sil_Score', 'Strict_neg_log_p', 'Strict_Enrichment'])

        for p_thresh in p_thresh_values:
            for alpha in lasso_alphas:
                print(f"\n>>> [{cancer_name}] 正在测试 P_Thresh = {p_thresh}, Lasso Alpha = {alpha} ...")
                try:
                    SURVIVE_SELECT = survive_select(survive, flatten, p_thresh=p_thresh, lasso_alpha=alpha)
                    num_features = SURVIVE_SELECT.shape[1]
                    print(f"    [*] P_Thresh={p_thresh}, Alpha={alpha} 筛选后剩余特征数: {num_features}")
                    
                    if num_features < 2:
                        print(f"    ⚠️ Lasso 特征未达标 (仅 {num_features} 个)，记录失败配置并跳过聚类！")
                        # 将失败记录显式写入 CSV，指标列采用 -1 占位
                        with open(cancer_detail_csv, mode='a', newline='') as f:
                            csv.writer(f).writerow([timestamp, p_thresh, alpha, -1, num_features, -1, -1, -1])
                        continue # 中断当前参数组合，跳过后续聚类计算与 R 语言评估

                    cp = ClusterProcessor(SURVIVE_SELECT, survive)

                    for k in k_values:
                        try:
                            # clusters = cp.KmeansCluster(k)
                            if cluster_method == 'kmeans':
                                clusters = cp.KmeansCluster(k)
                            elif cluster_method == 'srsc':
                                clusters = cp.SelfReinforcedSubspaceCluster(
                                    nclusters=k,
                                    alpha=1.0,
                                    eta=min(8, max(3, SURVIVE_SELECT.shape[0] // 20)),
                                    gamma=min(5, max(2, SURVIVE_SELECT.shape[0] // 40)),
                                    sigma=0.6,
                                    init_topk=min(10, max(3, SURVIVE_SELECT.shape[0] - 1)),
                                    n_refine_iters=3,
                                    random_state=42
                                )
                            else:
                                raise ValueError(f"Unsupported cluster_method: {cluster_method}")

                            sil_score = silhouette_score(SURVIVE_SELECT, clusters)

                            
                            df_for_r = pd.DataFrame({
                                'sample_name': SURVIVE_SELECT.index, 
                                'label': clusters,
                                'days': survive['days'].values,     # 强制传入对齐后的生存期
                                'status': survive['status'].values  # 强制传入对齐后的状态
                            })
                            
                            r_res = analyze_clusters_r(df_for_r, cancer_name)
                            strict_neg_log_p = float(r_res[0][0])
                            strict_enrichment = int(r_res[1][0])

                            with open(cancer_detail_csv, mode='a', newline='') as f:
                                csv.writer(f).writerow([timestamp, p_thresh, alpha, k, num_features, sil_score, strict_neg_log_p, strict_enrichment])
                                
                        except Exception as e:
                            pass

                except Exception as e:
                    print(f"[{cancer_name}] P_Thresh={p_thresh}, Alpha={alpha} 报错: {e}")

        # ================= 7. 挑选 SOTA 并记录 =================
        if not os.path.exists(cancer_detail_csv):
            continue

        res_df = pd.read_csv(cancer_detail_csv, on_bad_lines='skip')
        res_df = res_df[res_df['Timestamp'] == timestamp].dropna(subset=['P_Thresh', 'Lasso_Alpha'])
        
        if res_df.empty:
            continue
            
        # 统一转为数值型，防止解析错误
        res_df['Strict_Enrichment'] = pd.to_numeric(res_df['Strict_Enrichment'], errors='coerce').fillna(0)
        res_df['Strict_neg_log_p'] = pd.to_numeric(res_df['Strict_neg_log_p'], errors='coerce').fillna(0)
        res_df['Selected_Features'] = pd.to_numeric(res_df['Selected_Features'], errors='coerce').fillna(0)

        # 核心逻辑：剔除 Selected_Features < 2 的阵亡记录
        valid_features_df = res_df[res_df['Selected_Features'] >= 2]
        
        if valid_features_df.empty:
            print(f"\n❌ 严重阻断：{cancer_name} 的所有超参数组合均未通过特征筛选！不生成最终聚类 CSV。")
            continue # 彻底跳过当前癌症的数据导出

        # 从有效特征记录中挑选最优
        valid_res = valid_features_df[valid_features_df['Strict_Enrichment'] > 0]
        if valid_res.empty:
            best_res = valid_features_df.sort_values(by='Strict_neg_log_p', ascending=False).iloc[0]
        else:
            best_res = valid_res.sort_values(by='Strict_neg_log_p', ascending=False).iloc[0]
        
        best_k = int(best_res['K'])
        best_alpha = best_res['Lasso_Alpha']
        best_p_thresh = float(best_res['P_Thresh'])
        best_neg_log_p = float(best_res['Strict_neg_log_p'])
        best_enrichment = int(best_res['Strict_Enrichment'])
        best_features = int(best_res['Selected_Features'])
        best_sil = float(best_res['Sil_Score'])

        print(f"\n🏆 【{cancer_name} 本次最佳结果】 P_Thresh: {best_p_thresh}, Alpha: {best_alpha}, K: {best_k}, -logP: {best_neg_log_p:.4f}, 富集: {best_enrichment}")
        
        with open(summary_csv, mode='a', newline='') as f:
            csv.writer(f).writerow([cancer_name, best_p_thresh, best_alpha, best_k, best_features, best_sil, best_neg_log_p, best_enrichment])

        # 最终聚类保存 (因为上方已做拦截，此处提取的特征数必定 >= 2，删除了原有的强行替换 flatten 代码)
        FINAL_SURVIVE = survive_select(survive, flatten, p_thresh=best_p_thresh, lasso_alpha=best_alpha)
            
        final_cp = ClusterProcessor(FINAL_SURVIVE, survive)
        
        # final_clusters = final_cp.KmeansCluster(best_k)

        if cluster_method == 'kmeans':
            final_clusters = final_cp.KmeansCluster(best_k)
        elif cluster_method == 'srsc':
            final_clusters = final_cp.SelfReinforcedSubspaceCluster(
                nclusters=best_k,
                alpha=1.0,
                eta=min(8, max(3, FINAL_SURVIVE.shape[0] // 20)),
                gamma=min(5, max(2, FINAL_SURVIVE.shape[0] // 40)),
                sigma=0.6,
                init_topk=min(10, max(3, FINAL_SURVIVE.shape[0] - 1)),
                n_refine_iters=3,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported cluster_method: {cluster_method}")


        
        name_suffix = f"neglogp={best_neg_log_p:.4f}_enrichment={best_enrichment}_pthresh={best_p_thresh}_alpha={best_alpha}_k={best_k}"

        clean_df = pd.DataFrame({'sample_name': FINAL_SURVIVE.index, 'label': final_clusters})
        clean_df.to_csv(f"{cancer_out_dir}/{cancer_name}_{name_suffix}_{timestamp}_Clean.csv", index=False)

        clinical_df = survive.copy()
        clinical_df['Cluster'] = final_clusters
        clinical_df.to_csv(f"{cancer_out_dir}/{cancer_name}_{name_suffix}_{timestamp}_Clinical.csv", index=True, index_label='sample_name')

    except Exception as e:
        import traceback
        print(f"❌❌❌ 严重错误：{cancer_name} 数据集处理失败，已跳过！")
        print(f"报错详情：{traceback.format_exc()}")

    finally:
        tf.keras.backend.clear_session()

print(f"\n🎉🎉 全剧终！所有数据集跑完！请去查看 {summary_csv}")