import os

cancer_name = "UVM"

# os.environ['R_HOME'] = '/public/home/huangyunmin/miniconda3/lib/R'
# os.environ['PATH'] = '/public/home/huangyunmin/miniconda3/bin:' + os.environ.get('PATH', '')

os.makedirs(f'sorted_data', exist_ok=True)
os.makedirs(f'features', exist_ok=True)

# 把原来的 os.makedirs('result', exist_ok=True) 删掉
# 替换成你的绝对路径：
result_dir = '/public/home/huangyunmin/cancer_cmp/CA-CAE-main/CA-CAE-main/CACAE/result'
os.makedirs(result_dir, exist_ok=True)

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import torch
import pandas as pd
pd.Series.iteritems = pd.Series.items  
pd.DataFrame.iteritems = pd.DataFrame.items

# ================= Pandas 降级兼容补丁 =================
# 补丁 1：欺骗老版本的 lifelines 库，修复 iteritems
pd.Series.iteritems = pd.Series.items  
pd.DataFrame.iteritems = pd.DataFrame.items

# 补丁 2：拦截并删除已经被 Pandas 2.0 废弃的 datetime_is_numeric 参数
original_describe = pd.DataFrame.describe
def patched_describe(self, *args, **kwargs):
    kwargs.pop('datetime_is_numeric', None) # 如果有这个参数，就偷偷拿掉
    return original_describe(self, *args, **kwargs)
pd.DataFrame.describe = patched_describe
# =======================================================
import warnings
warnings.filterwarnings("ignore")

path2 = f"cancer_cmp/CA-CAE-main/CA-CAE-main/CACAE/Data/fea/{cancer_name}/miRNA.fea"
path3 = f"cancer_cmp/CA-CAE-main/CA-CAE-main/CACAE/Data/fea/{cancer_name}/rna.fea"
path4 = f"cancer_cmp/CA-CAE-main/CA-CAE-main/CACAE/Data/fea/{cancer_name}/meth.fea"
path5 = f"cancer_cmp/CA-CAE-main/CA-CAE-main/CACAE/Data/clinic/{cancer_name}.clinic"
path_cn = f"cancer_cmp/CA-CAE-main/CA-CAE-main/CACAE/Data/fea/{cancer_name}/CN.fea" # 增加 CN 路径



miRNA = pd.read_table(path2, sep=',', index_col=0).T
RNA = pd.read_table(path3, sep=',', index_col=0).T
Meth = pd.read_table(path4, sep=',', index_col=0).T
survive = pd.read_table(path5, sep=',', index_col=0)
CN = pd.read_table(path_cn, sep=',', index_col=0).T # 读取 CN 数据



from CACAE import Model, Process
from CACAE.utils import ClusterProcessor, do_km_plot
from CACAE.Survive_select import survive_select


RNA_processor = Process.DataProcessor(RNA)
RNA = RNA_processor.sort_corr(3300)
RNA.to_csv(os.path.join(f'sorted_data', f'{cancer_name}_sorted_mRNA.csv'), index=False)

miRNA_processor = Process.DataProcessor(miRNA)
miRNA = miRNA_processor.sort_corr(400)
miRNA.to_csv(os.path.join(f'sorted_data', f'{cancer_name}_sorted_miRNA.csv'), index=False)

Meth_processor = Process.DataProcessor(Meth)
Meth = Meth_processor.sort_corr(3200)
Meth.to_csv(os.path.join(f'sorted_data', f'{cancer_name}_sorted_Meth.csv'), index=False)

CN_processor = Process.DataProcessor(CN)
CN = CN_processor.sort_corr(3200)
CN.to_csv(os.path.join(f'sorted_data', f'{cancer_name}_sorted_CN.csv'), index=False)


RNA_model = Model.CACAE(RNA.shape[1])
miRNA_model = Model.CACAE(miRNA.shape[1])
Meth_model = Model.CACAE(Meth.shape[1])
CN_model = Model.CACAE(CN.shape[1]) # 增加 CN 模型
RNA_model.fit(RNA)
miRNA_model.fit(miRNA)
Meth_model.fit(Meth)
CN_model.fit(CN) # 训练 CN 模型



RNA_feature = pd.DataFrame(RNA_model.extract_feature(RNA), columns=[f"{cancer_name}_RNA_feat_{i}" for i in range(RNA_model.extract_feature(RNA).shape[1])])
miRNA_feature = pd.DataFrame(miRNA_model.extract_feature(miRNA), columns=[f"{cancer_name}_miRNA_feat_{i}" for i in range(miRNA_model.extract_feature(miRNA).shape[1])])
Meth_feature = pd.DataFrame(Meth_model.extract_feature(Meth), columns=[f"{cancer_name}_Meth_feat_{i}" for i in range(Meth_model.extract_feature(Meth).shape[1])])
CN_feature = pd.DataFrame(CN_model.extract_feature(CN), columns=[f"CN_feat_{i}" for i in range(CN_model.extract_feature(CN).shape[1])])




flatten = pd.concat([RNA_feature, miRNA_feature, Meth_feature, CN_feature], axis=1)
# SURVIVE_SELECT = survive_select(survive, flatten, 0.05)

# ================= 终极防空值与对齐补丁 (完美修复版) =================

# 🌟 核心修复：把丢失的患者 ID 从原始组学数据里给 flatten 找回来！
flatten.index = RNA.index 

print(f"👉 修复后 Survive 的前3个ID: {survive.index[:3].tolist()}")
print(f"👉 修复后 Flatten 的前3个ID: {flatten.index[:3].tolist()}")

# 1. 暴力清洗：统一转大写 -> 换横杠 -> 截取前 12 位
survive.index = survive.index.astype(str).str.upper().str.replace('.', '-').str[:12]
flatten.index = flatten.index.astype(str).str.upper().str.replace('.', '-').str[:12]

# 2. 去重（防止同一个病人切了两个病理片导致重复）
survive = survive[~survive.index.duplicated(keep='first')]
flatten = flatten[~flatten.index.duplicated(keep='first')]

# 3. 严格取交集
common_samples = survive.index.intersection(flatten.index)
survive = survive.loc[common_samples]
flatten = flatten.loc[common_samples]

print(f"[*] ✅ 对齐完成，共有 {len(common_samples)} 个样本进入生存筛选阶段！")

# 4. 暴力兜底防 NaN
flatten = flatten.fillna(0)
survive = survive.fillna(0)
# ========================================================

# =============================================================

import csv
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 引入 R 语言接口
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

# 预先加载并编译你的 R 语言脚本
print("\n📦 正在加载 R 语言严格评估模块...")
r_script_path = '/public/home/huangyunmin/cancer_cmp/CA-CAE-main/CA-CAE-main/CACAE/strict_eval.R'
ro.r.source(r_script_path)
analyze_clusters_r = ro.globalenv['analyze_clusters']

# 1. 定义网格搜索的参数空间
lasso_alphas = [0.0001]  
k_values = [5]
# lasso_alphas = [0.01, 0.03, 0.05, 0.08, 0.1]  
# k_values = [2, 3, 4, 5, 6]


# 2. 准备保存结果的 CSV 文件 (增加了 Strict_neg_log_p 和 Strict_Enrichment)
csv_file = '/public/home/huangyunmin/cancer_cmp/CA-CAE-main/CA-CAE-main/CACAE/result/GridSearch_Results_Strict.csv'
os.makedirs(os.path.dirname(csv_file), exist_ok=True)
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Lasso_Alpha', 'K', 'Selected_Features', 'Sil_Score', 'Strict_neg_log_p', 'Strict_Enrichment'])

print("\n🚀 [开始网格搜索] 探索最佳参数组合并进行 R 语言严格检验...")

# 3. 开始双重循环搜索
for alpha in lasso_alphas:
    print(f"\n>>> 正在测试 Lasso Alpha = {alpha} ...")
    try:
        # 执行特征筛选
        SURVIVE_SELECT = survive_select(survive, flatten, p_thresh=0.05, lasso_alpha=alpha)
        num_features = SURVIVE_SELECT.shape[1]
        
        if num_features < 2:
            print("    [!] 保留特征数太少，跳过当前 Alpha。")
            continue

        # 初始化聚类器
        cp = ClusterProcessor(SURVIVE_SELECT, survive)

        for k in k_values:
            try:
                # 获取聚类标签
                clusters = cp.KmeansCluster(k)
                
                # 计算内部聚类指标 (如轮廓系数)
                sil_score = silhouette_score(SURVIVE_SELECT, clusters)
                
                # 🌟 核心：构造传递给 R 语言的 DataFrame
                df_for_r = pd.DataFrame({
                    'sample_name': SURVIVE_SELECT.index,
                    'label': clusters
                })
                
                # 🌟 调用 R 语言进行严格评估
                r_res = analyze_clusters_r(df_for_r, cancer_name)
                
                # 提取 R 返回的结果（这里用 [0][0] 防止 NamedList 报错，就是我们第一轮解决的问题）
                strict_neg_log_p = float(r_res[0][0])
                strict_enrichment = int(r_res[1][0])

                print(f"    --> K={k} | Features={num_features} | Sil_Score={sil_score:.4f} | R_neg_log_p={strict_neg_log_p:.4f} | R_Enrich={strict_enrichment}")

                # 实时追加写入 CSV
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([alpha, k, num_features, sil_score, strict_neg_log_p, strict_enrichment])
                    
            except Exception as e:
                print(f"    [X] 测试 K={k} 时的评估报错: {e}")

    except Exception as e:
        print(f"[X] Lasso Alpha={alpha} 报错: {e}")

# ================= 搜索完成，挑选 SOTA 结果 =================
print(f"\n🎉 网格搜索全部完成！结果已保存至: {csv_file}")

try:
    res_df = pd.read_csv(csv_file)
    
    # 【容错保护 1】：如果 CSV 是空的，说明所有的 Alpha 都太严苛了
    if res_df.empty:
        print("❌ 严重警告：所有的 Lasso Alpha 都过于严格，导致没有保留下足够的特征！")
        print("👉 建议：请在代码中进一步调小 lasso_alphas 的值（如加入 0.0005, 0.001 等）。")
    else:
        # 挑选逻辑：要求富集数 >= 1，然后取 neg_log_p 最大的
        valid_res = res_df[res_df['Strict_Enrichment'] >= 1]
        
        if valid_res.empty:
            print("⚠️ 警告：所有组合的严格富集数都为 0，只能退而求其次选择生存最显著的...")
            best_res = res_df.sort_values(by='Strict_neg_log_p', ascending=False).iloc[0]
        else:
            best_res = valid_res.sort_values(by='Strict_neg_log_p', ascending=False).iloc[0]
        
        print("\n🏆 【CA-CAE 全局最佳参数配置 (严苛标准)】 🏆")
        best_alpha = best_res['Lasso_Alpha']
        best_k = int(best_res['K'])
        print(f"最佳 Lasso Alpha: {best_alpha}")
        print(f"最佳 聚类数 (K) : {best_k}")
        print(f"保留特征数量    : int({best_res['Selected_Features']})")
        print(f"生存显著性 (-log p) : {best_res['Strict_neg_log_p']:.4f}")
        print(f"临床特征富集数  : int({best_res['Strict_Enrichment']})")
        
        # ================= 5. 自动使用最佳参数生成最终结果与画图 =================
        print(f"\n>>> 正在使用最佳参数 (Alpha={best_alpha}, K={best_k}) 重新生成最终聚类与 KM 曲线...")
        
        # 提取最佳特征
        FINAL_SURVIVE = survive_select(survive, flatten, p_thresh=0.05, lasso_alpha=best_alpha)
        FINAL_SURVIVE.to_csv(os.path.join(f'features', f'{cancer_name}_features_best.csv'))
        
        # 使用最佳 K 值聚类
        final_cp = ClusterProcessor(FINAL_SURVIVE, survive)
        final_cp.compute_indexes(best_k)
        p_value, final_clusters = final_cp.LogRankp(best_k)
        
        # 保存聚类标签
        save_path = f"/public/home/huangyunmin/cancer_cmp/CA-CAE-main/CA-CAE-main/CACAE/result/{cancer_name}_BEST_Cluster_K{best_k}_Strict.csv"
        final_cp.save_clusters(final_clusters, save_path)
        print(f"✅ 最佳聚类标签已保存至: {save_path}")

        # 调用最终画图 (保留你原来的 do_km_plot 调用)
        do_km_plot(survive, pvalue=p_value.p_value, cindex=None, cancer_type=cancer_name, model_name='CA-CAE')
        print("✅ KM 曲线已生成！")

except Exception as e:
    print("\n❌ 展示最佳结果或执行最终流程时出现致命错误:", e)