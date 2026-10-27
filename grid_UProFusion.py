# ==============================================================================
import os

# os.environ['R_HOME'] = '/public/home/huangyunmin/miniconda3/lib/R'
# os.environ['PATH'] = '/public/home/huangyunmin/miniconda3/bin:' + os.environ.get('PATH', '')
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import datetime
import csv
import warnings
import pandas as pd
import tensorflow as tf
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import rpy2.robjects as ro
import random
import numpy as np


# ================= 🌟 强制全局随机种子固定 🌟 =================
def setup_seed(seed=42):
    # 1. 固定 Python 核心的哈希与随机数
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    
    # 2. 固定 NumPy 随机数
    np.random.seed(seed)
    
    # 3. 固定 TensorFlow 随机数 (替代 PyTorch)
    tf.random.set_seed(seed)
    # 开启 TensorFlow 确定性操作 (防止 GPU 上的多线程浮点计算随机性)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # 4. 固定 R 语言随机数 (关键：用于约束 glmnet 交叉验证和严格评估脚本中的随机采样)
    ro.r(f'set.seed({seed})')
    
    print(f"🔒 全局随机种子已固定为: {seed} (覆盖 Python, NumPy, TensorFlow, R)")

# 立即调用生效
setup_seed(42)
# =================================================================

# 修改输出目录以区分该版本的运行结果
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
base_run_dir = f'/public/home/huangyunmin/cancer_cmp/CA-CAE-main/CA-CAE-main/CACAE/Uproject/Run_{timestamp}_Uproject'
os.makedirs(base_run_dir, exist_ok=True)


# ================= 🌟 必不可少的 Pandas 降级补丁 🌟 =================
pd.Series.iteritems = pd.Series.items  
pd.DataFrame.iteritems = pd.DataFrame.items
original_describe = pd.DataFrame.describe
def patched_describe(self, *args, **kwargs):
    kwargs.pop('datetime_is_numeric', None)
    return original_describe(self, *args, **kwargs)
pd.DataFrame.describe = patched_describe
# =================================================================


warnings.filterwarnings("ignore")



class ContrastiveUProjection(tf.keras.Model):
    """
    多视图对比学习 U-Projection 融合网络。
    采用 混合损失 (Hybrid Loss: InfoNCE + MSE) 防止量纲坍缩。
    """
    def __init__(self, target_dim, view_dims, temperature=0.5, alpha=1.0, beta=1.0):
        super(ContrastiveUProjection, self).__init__()
        self.num_views = len(view_dims)
        self.temperature = temperature
        self.alpha = alpha  # InfoNCE 权重
        self.beta = beta    # MSE 权重
        
        # 1. 映射层 (View-specific Mapping)
        self.map_layers = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(target_dim, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(alpha=0.1)
            ]) for _ in range(self.num_views)
        ]
        
        # 2. 解码层 (View-specific Decoders - 用于 MSE 锚定)
        self.decoders = [
            tf.keras.layers.Dense(view_dims[i], activation='linear') 
            for i in range(self.num_views)
        ]
        
        # 3. U-Projection 瓶颈残差结构 (Shared Fusion)
        latent_dim = max(target_dim // 2, 2)
        self.fusion_down = tf.keras.layers.Dense(latent_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.1))
        self.fusion_up = tf.keras.layers.Dense(target_dim, activation='linear')
        self.bn_out = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        mapped_views = [self.map_layers[i](inputs[i], training=training) for i in range(self.num_views)]
        
        # 共识特征 (Consensus)
        stacked = tf.stack(mapped_views, axis=0)
        consensus = tf.reduce_mean(stacked, axis=0)
        
        # U-Projection 提纯
        z = self.fusion_down(consensus)
        z = self.fusion_up(z)
        fused = self.bn_out(consensus + z, training=training)
        
        # 重构 (Reconstruction) - 确保传入后续 Loss 计算
        reconstructed_views = [self.decoders[i](fused) for i in range(self.num_views)]
        
        return mapped_views, fused, reconstructed_views

    def compute_infonce(self, h1, h2):
        h1 = tf.math.l2_normalize(h1, axis=1)
        h2 = tf.math.l2_normalize(h2, axis=1)
        
        pos_sim = tf.reduce_sum(h1 * h2, axis=1)
        pos_exp = tf.exp(pos_sim / self.temperature)
        
        sim_matrix = tf.matmul(h1, h2, transpose_b=True)
        all_exp = tf.exp(sim_matrix / self.temperature)
        sum_all_exp = tf.reduce_sum(all_exp, axis=1)
        
        return tf.reduce_mean(-tf.math.log(pos_exp / (sum_all_exp + 1e-8)))

    @tf.function
    def train_step(self, data):
        if isinstance(data, tuple) and len(data) == self.num_views:
            x = data
        elif isinstance(data, tuple) and len(data) == 1:
            x = data[0]
        else:
            x = data
            
        with tf.GradientTape() as tape:
            mapped_views, fused, reconstructed_views = self(x, training=True)
            info_nce_loss = 0.0
            mse_loss = 0.0
            
            # 1. 对比损失计算 (InfoNCE)
            for i in range(self.num_views):
                # 局部-局部对齐
                for j in range(i + 1, self.num_views):
                    info_nce_loss += self.compute_infonce(mapped_views[i], mapped_views[j])
                # 全局-局部对齐
                info_nce_loss += self.compute_infonce(fused, mapped_views[i])
                
            # 2. 重构损失计算 (MSE)
            for i in range(self.num_views):
                mse_loss += tf.reduce_mean(tf.square(x[i] - reconstructed_views[i]))
            
            # 混合总损失
            total_loss = (self.alpha * info_nce_loss) + (self.beta * mse_loss)
                    
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {"total_loss": total_loss, "info_nce": info_nce_loss, "mse": mse_loss}


# ================= 1. 全局配置与 R 语言初始化 =================
pandas2ri.activate()
print("📦 正在加载 R 语言严格评估模块...")
ro.r.source('/public/home/huangyunmin/cancer_cmp/CA-CAE-main/CA-CAE-main/CACAE/strict_eval.R')
analyze_clusters_r = ro.globalenv['analyze_clusters']

cancer_datasets = ['KIRC','BLCA','BRCA','PAAD','STAD','GBM','SKCM','UCEC','UVM','LUAD']
# cancer_datasets = ['GBM']

lasso_alphas = [0.01,0.05,0.1,0.5,1]
k_values = [2,3,4,5]


summary_csv = f'{base_run_dir}/PanCancer_SOTA_Summary.csv'
os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
with open(summary_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Cancer_Type', 'Best_Alpha', 'Best_K', 'Selected_Features', 'Sil_Score', 'Strict_neg_log_p', 'Strict_Enrichment'])

print(f"📂 本次实验所有结果将保存在独立文件夹: {base_run_dir}")


for cancer_name in cancer_datasets:
    print("\n" + "🚀"*20)
    print(f"   正在全自动处理数据集: {cancer_name}")
    print("🚀"*20)

    cancer_out_dir = f'/public/home/huangyunmin/cancer_cmp/CA-CAE-main/CA-CAE-main/CACAE/Uproject/{cancer_name}'
    os.makedirs(cancer_out_dir, exist_ok=True)

    try:
        
        path2 = f"Cancer_data/fea/{cancer_name}/miRNA.fea"
        path3 = f"Cancer_data/fea/{cancer_name}/rna.fea"
        path4 = f"Cancer_data/fea/{cancer_name}/meth.fea"
        path5 = f"Cancer_data/clinic/{cancer_name}.clinic"
        path_cn = f"Cancer_data/fea/{cancer_name}/CN.fea" 

        miRNA = pd.read_table(path2, sep=',', index_col=0).T
        RNA = pd.read_table(path3, sep=',', index_col=0).T
        Meth = pd.read_table(path4, sep=',', index_col=0).T
        survive = pd.read_table(path5, sep=',', index_col=0)
        CN = pd.read_table(path_cn, sep=',', index_col=0).T 

        # ==============================================================================
        total_train_samples = RNA.shape[0]
        
        print(f"\n[*] 侦测到当前癌症 {cancer_name} 的组学样本总数为: {total_train_samples}")
        print("⚙️ 正在自动分配自适应超参数...")
        
        if total_train_samples < 150:
            dynamic_batch_size, dynamic_epochs = 8, 200
        elif total_train_samples < 400:
            dynamic_batch_size, dynamic_epochs = 16, 300
        elif total_train_samples < 800:
            dynamic_batch_size, dynamic_epochs = 32, 500
        else:
            dynamic_batch_size, dynamic_epochs = 64, 500

        if total_train_samples < 100:
            dynamic_val_rate = 0.1
        elif total_train_samples < 300:
            dynamic_val_rate = 0.15
        else:
            dynamic_val_rate = 0.2

        print(f"  -> 分配结果: Batch Size = {dynamic_batch_size} | Val Rate = {dynamic_val_rate} | Epochs = {dynamic_epochs}\n")
        # ==============================================================================

        from CACAE import Model, Process
        from CACAE.utils import ClusterProcessor, do_km_plot
        from CACAE.Survive_select import survive_select

        # --- 2.2 特征降噪 ---
        RNA_processor = Process.DataProcessor(RNA)
        RNA = RNA_processor.sort_corr(3200)
        
        miRNA_processor = Process.DataProcessor(miRNA)
        miRNA = miRNA_processor.sort_corr(300)
        
        Meth_processor = Process.DataProcessor(Meth)
        Meth = Meth_processor.sort_corr(3100)
        
        CN_processor = Process.DataProcessor(CN)
        CN = CN_processor.sort_corr(3100)

        # ================= 🌟 核心修复：防止上游网络梯度爆炸 🌟 =================
        from sklearn.preprocessing import MinMaxScaler
        
        print("\n⚙️ 正在执行全局 Min-Max 归一化，防止极值引发 NaN Loss...")
        # 使用 MinMaxScaler 将所有特征严格约束到 [0,1] 区间
        scaler = MinMaxScaler()
        RNA = pd.DataFrame(scaler.fit_transform(RNA), index=RNA.index, columns=RNA.columns)
        miRNA = pd.DataFrame(scaler.fit_transform(miRNA), index=miRNA.index, columns=miRNA.columns)
        Meth = pd.DataFrame(scaler.fit_transform(Meth), index=Meth.index, columns=Meth.columns)
        CN = pd.DataFrame(scaler.fit_transform(CN), index=CN.index, columns=CN.columns)
        
        # --- 2.3 初始化并训练模型 ---
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

        # --- 2.4 提取特征 ---
        RNA_feature = pd.DataFrame(RNA_model.extract_feature(RNA), columns=[f"{cancer_name}_RNA_feat_{i}" for i in range(RNA_model.extract_feature(RNA).shape[1])])
        miRNA_feature = pd.DataFrame(miRNA_model.extract_feature(miRNA), columns=[f"{cancer_name}_miRNA_feat_{i}" for i in range(miRNA_model.extract_feature(miRNA).shape[1])])
        Meth_feature = pd.DataFrame(Meth_model.extract_feature(Meth), columns=[f"{cancer_name}_Meth_feat_{i}" for i in range(Meth_model.extract_feature(Meth).shape[1])])
        CN_feature = pd.DataFrame(CN_model.extract_feature(CN), columns=[f"CN_feat_{i}" for i in range(CN_model.extract_feature(CN).shape[1])])

        # ==============================================================================
        # 🌟 修改点：执行多视图对比学习 U-Projection 融合 🌟
        print("\n开始执行多视图对比 U-Projection 特征融合...")
        
        view_list = [
            tf.convert_to_tensor(RNA_feature.fillna(0).values, dtype=tf.float32),
            tf.convert_to_tensor(miRNA_feature.fillna(0).values, dtype=tf.float32),
            tf.convert_to_tensor(Meth_feature.fillna(0).values, dtype=tf.float32),
            tf.convert_to_tensor(CN_feature.fillna(0).values, dtype=tf.float32)
        ]


        num_views = len(view_list)
        view_dims = [v.shape[1] for v in view_list] # 获取原始各视图维度
        target_dim = int(np.mean(view_dims)) 
        
        # 2. 初始化对比学习融合模型 (传入 view_dims 供 MSE 使用)
        fusion_model = ContrastiveUProjection(
            target_dim=target_dim, 
            view_dims=view_dims, 
            temperature=0.5,
            alpha=1.0,  # 对比权重
            beta=10.0   # MSE 权重放大，强化方差保留
        )
        fusion_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        
        # 3. 传入列表进行自监督对比训练 
        # 🚨 核心修复：抛弃 dynamic_batch_size，强制使用全批量 (Full-Batch)
        # 以确保 InfoNCE 拥有最大的负样本池
        print(f"    正在通过 InfoNCE + MSE 混合损失进行联合对齐 (Target: {target_dim}D, Batch: Full)...")
        tf_dataset = tf.data.Dataset.from_tensor_slices(tuple(view_list)).batch(total_train_samples)
        fusion_model.fit(tf_dataset, epochs=150, verbose=0)
        
        # 4. 推理阶段：提取最终的共识融合特征 (fused)
        _, fused_tensor, _ = fusion_model(view_list, training=False)
        
        flatten = pd.DataFrame(
            fused_tensor.numpy(), 
            index=RNA_feature.index,
            columns=[f"UPro_Fused_Feat_{i}" for i in range(target_dim)]
        )
        print(f"✅ U-Projection 混合对齐完成！最终用于 LASSO 筛选的特征维度: {flatten.shape}")
        # ==============================================================================

        # --- 2.5 临床 ID 对齐补丁 ---
        flatten.index = RNA.index 

        print(f"👉 修复后 Survive 的前3个ID: {survive.index[:3].tolist()}")
        print(f"👉 修复后 Flatten 的前3个ID: {flatten.index[:3].tolist()}")

        survive.index = survive.index.astype(str).str.upper().str.replace('.', '-').str[:12]
        flatten.index = flatten.index.astype(str).str.upper().str.replace('.', '-').str[:12]

        survive = survive[~survive.index.duplicated(keep='first')]
        flatten = flatten[~flatten.index.duplicated(keep='first')]

        common_samples = survive.index.intersection(flatten.index)
        survive = survive.loc[common_samples]
        flatten = flatten.loc[common_samples]
        
        survive['days'] = pd.to_numeric(survive['days'], errors='coerce')
        survive['status'] = pd.to_numeric(survive['status'], errors='coerce')

        print(f"[*] ✅ 对齐完成，共有 {len(common_samples)} 个样本进入生存筛选阶段！")

        flatten = flatten.fillna(0)
        survive = survive.fillna(0)


        # ================= 3. 针对当前癌症开始多维网格搜索 (Multi-dimensional Grid Search) =================
        global_results_dir = '/public/home/huangyunmin/cancer_cmp/CA-CAE-main/CA-CAE-main/CACAE/Uproject'     
        cancer_detail_csv = f'{global_results_dir}/{cancer_name}_UPro_Global_GridSearch.csv'

        # 1. 定义预筛选阈值 (Pre-screening Threshold) 的搜索空间
        p_thresh_values = [0.01, 0.05, 0.1, 0.2]

        # 2. 判断文件是否已经存在
        file_exists = os.path.isfile(cancer_detail_csv)
        
        # 3. 使用 'a' (Append) 模式打开文件。
        with open(cancer_detail_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                # 🚨 修正点 1：表头新增 'P_Thresh' 列，共计 8 列，确保数据严格对齐 (Data Alignment)
                writer.writerow(['Timestamp', 'P_Thresh', 'Lasso_Alpha', 'K', 'Selected_Features', 'Sil_Score', 'Strict_neg_log_p', 'Strict_Enrichment'])
        # =======================================================================

        # 4. 增加外层循环，执行三维超参数遍历 (P_Thresh -> Alpha -> K)
        for p_thresh in p_thresh_values:
            for alpha in lasso_alphas:
                print(f"\n>>> [{cancer_name}] 正在测试 P_Thresh = {p_thresh}, Lasso Alpha = {alpha} ...")
                try:
                    # 🚨 修正点 2：将写死的 p_thresh=0.2 替换为循环变量 p_thresh
                    SURVIVE_SELECT = survive_select(survive, flatten, p_thresh=p_thresh, lasso_alpha=alpha)
                    num_features = SURVIVE_SELECT.shape[1]
                    print(f"    [*] P_Thresh={p_thresh}, Alpha={alpha} 筛选后剩余特征数: {num_features}")
                    
                    if num_features < 2:
                        print(f"    ⚠️ Lasso 崩溃！触发保底机制：直接使用高级融合的全部 Flatten 特征 ({flatten.shape[1]} 个) 进行聚类！")
                        SURVIVE_SELECT = flatten
                        num_features = SURVIVE_SELECT.shape[1]

                    cp = ClusterProcessor(SURVIVE_SELECT, survive)

                    for k in k_values:
                        try:
                            clusters = cp.KmeansCluster(k)
                            sil_score = silhouette_score(SURVIVE_SELECT, clusters)
                            
                            df_for_r = pd.DataFrame({'sample_name': SURVIVE_SELECT.index, 'label': clusters})
                            r_res = analyze_clusters_r(df_for_r, cancer_name)
                            strict_neg_log_p = float(r_res[0][0])
                            strict_enrichment = int(r_res[1][0])

                            # 🚨 修正点 3：写入列表的第二项必须插入 p_thresh，确保 8 列数据与表头一一对应
                            with open(cancer_detail_csv, mode='a', newline='') as f:
                                csv.writer(f).writerow([timestamp, p_thresh, alpha, k, num_features, sil_score, strict_neg_log_p, strict_enrichment])
                                
                        except Exception as e:
                            pass

                except Exception as e:
                    print(f"[{cancer_name}] P_Thresh={p_thresh}, Alpha={alpha} 报错: {e}")


        # ================= 4. 挑选当前癌症 SOTA 并记录 (核心参数提取适配) =================
        # 注意：由于表头从 7 列变为了 8 列，下游提取最佳模型 (SOTA) 的逻辑也必须同步修改。
        if not os.path.exists(cancer_detail_csv):
            continue

        res_df = pd.read_csv(cancer_detail_csv, on_bad_lines='skip')
        res_df = res_df[res_df['Timestamp'] == timestamp]
        res_df = res_df.dropna(subset=['Lasso_Alpha', 'K', 'P_Thresh'])
        
        if res_df.empty:
            continue
            
        res_df['Strict_Enrichment'] = pd.to_numeric(res_df['Strict_Enrichment'], errors='coerce').fillna(0)
        res_df['Strict_neg_log_p'] = pd.to_numeric(res_df['Strict_neg_log_p'], errors='coerce').fillna(0)

        valid_res = res_df[res_df['Strict_Enrichment'] > 0]
        if valid_res.empty:
            best_res = res_df.sort_values(by='Strict_neg_log_p', ascending=False).iloc[0]
        else:
            best_res = valid_res.sort_values(by='Strict_neg_log_p', ascending=False).iloc[0]
        
        best_k = int(best_res['K'])
        best_alpha = best_res['Lasso_Alpha']
        best_p_thresh = float(best_res['P_Thresh']) # 🚨 新增：提取最佳 P_Thresh
        best_neg_log_p = float(best_res['Strict_neg_log_p'])
        best_enrichment = int(best_res['Strict_Enrichment'])

        # 🚨 提取最佳参数并执行最终筛选
        FINAL_SURVIVE = survive_select(survive, flatten, p_thresh=best_p_thresh, lasso_alpha=best_alpha)
        
        # ================= 修复核心：补全缺失的实例化与特征保底机制 =================
        if FINAL_SURVIVE.shape[1] < 2:
            print(f"    ⚠️ 最终保存阶段 Lasso 特征不足，触发保底机制！")
            FINAL_SURVIVE = flatten
            
        final_cp = ClusterProcessor(FINAL_SURVIVE, survive)
        final_clusters = final_cp.KmeansCluster(best_k)
        # =========================================================================
        
        # ================= 🌟 动态重命名并保存 =================
        name_suffix = f"neglogp={best_neg_log_p:.4f}_enrichment={best_enrichment}_pthresh={best_p_thresh}_alpha={best_alpha}_k={best_k}"

        clean_df = pd.DataFrame({
            'sample_name': FINAL_SURVIVE.index,
            'label': final_clusters
        })
        clean_csv_path = f"{cancer_out_dir}/{cancer_name}_{name_suffix}_{timestamp}_Clean.csv"
        clean_df.to_csv(clean_csv_path, index=False)

        clinical_df = survive.copy()
        clinical_df['Cluster'] = final_clusters
        clinical_csv_path = f"{cancer_out_dir}/{cancer_name}_{name_suffix}_{timestamp}_Clinical.csv"
        clinical_df.to_csv(clinical_csv_path, index=True, index_label='sample_name')

    except Exception as e:
        import traceback
        print(f"❌❌❌ 严重错误：{cancer_name} 数据集处理失败，已跳过！")
        print(f"报错详情：{traceback.format_exc()}")

    finally:
        tf.keras.backend.clear_session()

print(f"\n🎉🎉 全剧终！所有数据集跑完！请去查看 {summary_csv}") 