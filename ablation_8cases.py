# ==============================================================================
# 脚本名称: ablation_8cases.py
# 核心逻辑: 全面消融实验 (Graph Denoise, U-Projection, SRSC vs K-means)
# ==============================================================================
import os
import gc
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

# ================= R语言环境配置 (已注释) =================
# os.environ['R_HOME'] = '/public/home/tanjingbo/miniconda3/envs/r423/lib/R'
# os.environ['PATH'] = '/public/home/tanjingbo/miniconda3/envs/r423/bin:' + os.environ.get('PATH', '')
# os.environ['LD_LIBRARY_PATH'] = '/public/home/tanjingbo/miniconda3/envs/r423/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro

warnings.filterwarnings("ignore")
pandas2ri.activate()

# ================= 1. 全局种子与环境 =================
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

pd.Series.iteritems = pd.Series.items  
pd.DataFrame.iteritems = pd.DataFrame.items
original_describe = pd.DataFrame.describe
def patched_describe(self, *args, **kwargs):
    kwargs.pop('datetime_is_numeric', None)
    return original_describe(self, *args, **kwargs)
pd.DataFrame.describe = patched_describe

# 加载 R 评估模块 (路径已替换为 tanjingbo)
print("📦 正在加载 R 语言严格评估模块...")
ro.r.source('/public/home/tanjingbo/CA-CAE-main/CACAE/strict_eval.R')
analyze_clusters_r = ro.globalenv['analyze_clusters']

# ================= 2. 网络组件定义 =================
class SSLNMVC_CommonBranch(tf.keras.Model):
    def __init__(self, hidden_dim, mlp_layers=1, use_bn=False):
        super(SSLNMVC_CommonBranch, self).__init__()
        self.hidden_dim = hidden_dim
        latent_dim = hidden_dim // 2 

        self.map_layer = tf.keras.Sequential(name="map_layer")
        self.map_layer.add(tf.keras.layers.Dense(hidden_dim, use_bias=True))
        if use_bn:
            self.map_layer.add(tf.keras.layers.BatchNormalization())
        self.map_layer.add(tf.keras.layers.ReLU())

        self.fusion = tf.keras.Sequential(name="fusion")
        self.fusion.add(tf.keras.layers.Dense(latent_dim, use_bias=False))
        if use_bn:
            self.fusion.add(tf.keras.layers.BatchNormalization())
        self.fusion.add(tf.keras.layers.ReLU())
        
        for _ in range(mlp_layers):
            self.fusion.add(tf.keras.layers.Dense(latent_dim, use_bias=False))
            if use_bn:
                self.fusion.add(tf.keras.layers.BatchNormalization())
            self.fusion.add(tf.keras.layers.ReLU())
            
        self.fusion.add(tf.keras.layers.Dense(hidden_dim, use_bias=False))

    def call(self, h_concat, training=False):
        z = self.map_layer(h_concat, training=training)
        res = z  
        z_fused = self.fusion(z, training=training)
        commonZ = z_fused + res
        return commonZ

def orthogonality_loss(z_common, z_specific_list):
    loss = 0.0
    z_common_norm = tf.math.l2_normalize(z_common, axis=0)
    for z_spec in z_specific_list:
        z_spec_norm = tf.math.l2_normalize(z_spec, axis=0)
        corr_matrix = tf.matmul(z_common_norm, z_spec_norm, transpose_a=True)
        loss += tf.reduce_mean(tf.square(corr_matrix))
    return loss

def topological_denoising(df_features, k=5):
    A = kneighbors_graph(df_features.values, n_neighbors=k, mode='connectivity', include_self=False)
    A = A.maximum(A.T) 
    I = sp.eye(A.shape[0])
    A_tilde = A + I
    
    D_tilde = np.array(A_tilde.sum(axis=1)).flatten()
    D_inv_sqrt = np.power(D_tilde, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    D_inv_sqrt_mat = sp.diags(D_inv_sqrt)
    
    S = D_inv_sqrt_mat.dot(A_tilde).dot(D_inv_sqrt_mat)
    X_smooth = S.dot(df_features.values)
    return pd.DataFrame(X_smooth, index=df_features.index, columns=df_features.columns)

# ================= 3. 用户参数配置区 =================
optimal_params = {
    # "KIRC": {"p_thresh": 0.005, "alpha": 0.5, "k": 4},
    # "BLCA": {"p_thresh": 0.01, "alpha": 1.0, "k": 5},
    # "BRCA": {"p_thresh": 0.15, "alpha": 1.0, "k": 4},
    # "PAAD": {"p_thresh": 0.15, "alpha": 1.0, "k": 4},
    # "STAD": {"p_thresh": 0.01, "alpha": 0.5, "k": 3},
    # "SKCM": {"p_thresh": 0.01, "alpha": 0.5, "k": 3},
    # "LUAD": {"p_thresh": 0.15, "alpha": 1, "k": 5},
    "CHOL": {"p_thresh": 0.005, "alpha": 0.5, "k": 2},
    # "LUSC": {"p_thresh": 0.005, "alpha": 0.5, "k": 2},
    # "SARC": {"p_thresh": 0.005, "alpha": 0.01, "k": 5},
    # "CESC": {"p_thresh": 0.01, "alpha": 0.5, "k": 3},
    # "COAD": {"p_thresh": 0.3, "alpha": 0.01, "k": 4},
    # "DLBC": {"p_thresh": 0.3, "alpha": 0.5, "k": 2},
    # "LIHC": {"p_thresh": 0.15, "alpha": 1.0, "k": 5}
}

# 8种消融实验配置矩阵 (2^3 = 8 cases)
ablation_configs = [
    {"name": "1_Full_Model",       "use_denoise": True,  "use_uproject": True,  "cluster_method": "srsc"},
    {"name": "2_w/o_Denoise",      "use_denoise": False, "use_uproject": True,  "cluster_method": "srsc"},
    {"name": "3_w/o_UProject",     "use_denoise": True,  "use_uproject": False, "cluster_method": "srsc"},
    {"name": "4_w/o_SRSC",         "use_denoise": True,  "use_uproject": True,  "cluster_method": "kmeans"},
    {"name": "5_w/o_GD_UP",        "use_denoise": False, "use_uproject": False, "cluster_method": "srsc"},
    {"name": "6_w/o_GD_SRSC",      "use_denoise": False, "use_uproject": True,  "cluster_method": "kmeans"},
    {"name": "7_w/o_UP_SRSC",      "use_denoise": True,  "use_uproject": False, "cluster_method": "kmeans"},
    {"name": "8_Baseline(w/o_All)","use_denoise": False, "use_uproject": False, "cluster_method": "kmeans"}
]

# ================= 4. 环境初始化 =================
cancer_datasets = list(optimal_params.keys())
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
# 输出目录改为 tanjingbo
base_run_dir = f'/public/home/tanjingbo/CA-CAE-main/CACAE/Ablation_8Cases_{timestamp}'
os.makedirs(base_run_dir, exist_ok=True)

summary_csv = f'{base_run_dir}/Ablation_8Cases_Summary.csv'
with open(summary_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'Cancer_Type', 'Config_Name', 'GraphDenoise', 'UProject', 'ClusterMethod', 
        'Selected_Features', 'Sil_Score', 'Strict_neg_log_p', 'Strict_Enrichment'
    ])

from CACAE import Model, Process
from CACAE.utils import ClusterProcessor
from CACAE.Survive_select import survive_select

# ================= 5. 主循环执行流 =================
for cancer_name in cancer_datasets:
    print(f"\n{'='*50}\n🚀 开始执行数据集消融: {cancer_name}\n{'='*50}")
    
    # 提取当前数据集最佳超参
    params = optimal_params[cancer_name]
    best_p = params["p_thresh"]
    best_alpha = params["alpha"]
    best_k = params["k"]

    for config in ablation_configs:
        config_name = config["name"]
        use_denoise = config["use_denoise"]
        use_uproject = config["use_uproject"]
        cluster_method = config["cluster_method"]
        print(f"\n>>> 正在运行配置: [{config_name}] (GD={use_denoise}, UP={use_uproject}, Cluster={cluster_method})")

        try:
            # --- 5.1 数据加载与基础处理 ---
            miRNA = pd.read_table(f"/public/home/tanjingbo/CA-CAE-main/CACAE/Data/fea/{cancer_name}/miRNA.fea", sep=',', index_col=0).T
            RNA = pd.read_table(f"/public/home/tanjingbo/CA-CAE-main/CACAE/Data/fea/{cancer_name}/rna.fea", sep=',', index_col=0).T
            Meth = pd.read_table(f"/public/home/tanjingbo/CA-CAE-main/CACAE/Data/fea/{cancer_name}/meth.fea", sep=',', index_col=0).T
            survive = pd.read_table(f"/public/home/tanjingbo/CA-CAE-main/CACAE/Data/clinic/{cancer_name}.clinic", sep=',', index_col=0)
            CN = pd.read_table(f"/public/home/tanjingbo/CA-CAE-main/CACAE/Data/fea/{cancer_name}/CN.fea", sep=',', index_col=0).T 

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

            # --- 基础特征筛选与数据插补 ---
            RNA = Process.DataProcessor(RNA).sort_corr(3200).fillna(0)
            miRNA = Process.DataProcessor(miRNA).sort_corr(300).fillna(0)
            Meth = Process.DataProcessor(Meth).sort_corr(3100).fillna(0)
            CN = Process.DataProcessor(CN).sort_corr(3100).fillna(0)

            # --- 强制归一化防御 ---
            scaler = MinMaxScaler()
            RNA = pd.DataFrame(scaler.fit_transform(RNA), index=RNA.index, columns=RNA.columns)
            miRNA = pd.DataFrame(scaler.fit_transform(miRNA), index=miRNA.index, columns=miRNA.columns)
            Meth = pd.DataFrame(scaler.fit_transform(Meth), index=Meth.index, columns=Meth.columns)
            CN = pd.DataFrame(scaler.fit_transform(CN), index=CN.index, columns=CN.columns)

            # --- 5.3 消融分支 1: 拓扑降噪 ---
            if use_denoise:
                k_neighbors = 3 if total_train_samples < 200 else 5
                RNA = topological_denoising(RNA, k=k_neighbors)
                miRNA = topological_denoising(miRNA, k=k_neighbors)
                Meth = topological_denoising(Meth, k=k_neighbors)
                CN = topological_denoising(CN, k=k_neighbors)
                print("    [+] 已开启 Graph Denoising")
            else:
                print("    [-] 已关闭 Graph Denoising")

            # --- 5.4 CA-CAE 私有特征提取 ---
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

            RNA_feature = pd.DataFrame(z_rna, columns=[f"{cancer_name}_RNA_{i}" for i in range(z_rna.shape[1])])
            miRNA_feature = pd.DataFrame(z_mirna, columns=[f"{cancer_name}_miRNA_{i}" for i in range(z_mirna.shape[1])])
            Meth_feature = pd.DataFrame(z_meth, columns=[f"{cancer_name}_Meth_{i}" for i in range(z_meth.shape[1])])
            CN_feature = pd.DataFrame(z_cn, columns=[f"CN_{i}" for i in range(z_cn.shape[1])])
            
            # --- 5.5 消融分支 2: U-Projection ---
            if use_uproject:
                print("    [+] 已开启 U-Projection 共识特征提取")
                z_spec_list = [tf.cast(z_rna, tf.float32), tf.cast(z_mirna, tf.float32), tf.cast(z_meth, tf.float32), tf.cast(z_cn, tf.float32)]
                h_concat = tf.concat(z_spec_list, axis=1)

                hidden_dim = h_concat.shape[1] // 2 
                common_branch = SSLNMVC_CommonBranch(hidden_dim=hidden_dim, mlp_layers=1, use_bn=False)
                common_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
                
                dataset = tf.data.Dataset.from_tensor_slices((h_concat, z_rna, z_mirna, z_meth, z_cn)).batch(dynamic_batch_size, drop_remainder=True)
                
                for epoch in range(50):
                    for batch_h, b_rna, b_mirna, b_meth, b_cn in dataset:
                        with tf.GradientTape() as tape:
                            commonZ = common_branch(batch_h, training=True)
                            batch_z_spec_list = [tf.cast(b_rna, tf.float32), tf.cast(b_mirna, tf.float32), tf.cast(b_meth, tf.float32), tf.cast(b_cn, tf.float32)]
                            loss_ortho = orthogonality_loss(commonZ, batch_z_spec_list)
                        grads = tape.gradient(loss_ortho, common_branch.trainable_variables)
                        common_optimizer.apply_gradients(zip(grads, common_branch.trainable_variables))
                
                z_common_final = common_branch(h_concat).numpy()
                Common_feature = pd.DataFrame(z_common_final, columns=[f"CommonZ_{i}" for i in range(z_common_final.shape[1])])
                
                flatten = pd.concat([RNA_feature, miRNA_feature, Meth_feature, CN_feature, Common_feature], axis=1)
            else:
                print("    [-] 已关闭 U-Projection，直接进行特定特征拼接")
                flatten = pd.concat([RNA_feature, miRNA_feature, Meth_feature, CN_feature], axis=1)

            # --- 5.6 临床 ID 对齐补丁与生存清洗 ---
            flatten.index = RNA.index
            
            # 使用 regex=False 防护特殊字符，严格规避截断不对齐
            flatten.index = flatten.index.astype(str).str.upper().str.replace('.', '-', regex=False).str[:12]
            survive.index = survive.index.astype(str).str.upper().str.replace('.', '-', regex=False).str[:12]

            survive = survive[~survive.index.duplicated(keep='first')]
            flatten = flatten[~flatten.index.duplicated(keep='first')]
            
            common_samples = survive.index.intersection(flatten.index)
            survive = survive.loc[common_samples]
            flatten = flatten.loc[common_samples]
            
            survive['days'] = pd.to_numeric(survive['days'], errors='coerce')
            survive['status'] = pd.to_numeric(survive['status'], errors='coerce')
            
            survive = survive.dropna(subset=['days', 'status'])
            survive = survive[survive['days'] > 0]
            flatten = flatten.fillna(0)
            
            common_samples_final = survive.index.intersection(flatten.index)
            survive = survive.loc[common_samples_final]
            flatten = flatten.loc[common_samples_final]
            print(f"[*] ✅ 对齐与清洗完成，共有 {len(common_samples_final)} 个有效生存样本进入筛选！")

            FINAL_SURVIVE = survive_select(survive, flatten, p_thresh=best_p, lasso_alpha=best_alpha)
            num_features = FINAL_SURVIVE.shape[1]
            
            # --- 5.7 动态消融分支 3: SRSC vs K-Means ---
            if num_features < 2:
                print(f"    ⚠️ Lasso 特征未达标 (仅 {num_features} 个)，本配置评估失效！")
                with open(summary_csv, mode='a', newline='') as f:
                    csv.writer(f).writerow([
                        cancer_name, config_name, use_denoise, use_uproject, cluster_method, 
                        num_features, -1, -1, -1
                    ])
                continue

            cp = ClusterProcessor(FINAL_SURVIVE, survive)
            
            # 依据消融配置动态分流聚类算法
            if cluster_method == 'kmeans':
                final_clusters = cp.KmeansCluster(best_k)
            elif cluster_method == 'srsc':
                final_clusters = cp.SelfReinforcedSubspaceCluster(
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

            sil_score = silhouette_score(FINAL_SURVIVE, final_clusters)
            
            # 打包数据，调用 R 进行评估
            df_for_r = pd.DataFrame({
                'sample_name': FINAL_SURVIVE.index, 
                'label': final_clusters,
                'days': survive['days'].values,
                'status': survive['status'].values
            })
            r_res = analyze_clusters_r(df_for_r, cancer_name)
            
            strict_neg_log_p = float(r_res[0][0])
            strict_enrichment = int(r_res[1][0])

            print(f"    ✅ 评估完成: 特征数={num_features}, Sil={sil_score:.4f}, -logP={strict_neg_log_p:.4f}, Enrich={strict_enrichment}")
            
            with open(summary_csv, mode='a', newline='') as f:
                csv.writer(f).writerow([
                    cancer_name, config_name, use_denoise, use_uproject, cluster_method, 
                    num_features, sil_score, strict_neg_log_p, strict_enrichment
                ])

        except Exception as e:
            import traceback
            print(f"    ❌ 发生异常: {e}")
            with open(summary_csv, mode='a', newline='') as f:
                csv.writer(f).writerow([
                    cancer_name, config_name, use_denoise, use_uproject, cluster_method, 
                    "ERROR", "ERROR", "ERROR", "ERROR"
                ])
        
        finally:
            tf.keras.backend.clear_session()
            gc.collect()

print(f"\n🎉 8种消融情况自动化完毕！汇总结果已保存至: {summary_csv}")