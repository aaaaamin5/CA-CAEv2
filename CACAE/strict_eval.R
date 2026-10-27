# conda activate r423
# python /public/home/tanjingbo/CA-CAE-main/grid_GD_dual.py



# 1. 首先定义所有必要的函数
library(survival)
library(parallel)

# 读取临床数据
get.clinical.data <- function(sub_name) {
  # 【唯一修改处】：把 D 盘路径换成了你的服务器路径
  clinic_path <- paste0('/public/home/tanjingbo/CA-CAE-main/CACAE/Data/clinic/', sub_name, '.clinic')
  if (!file.exists(clinic_path)) {
    stop(paste("找不到临床文件:", clinic_path))
  }
  clinic_data <- read.csv(clinic_path, header = TRUE, stringsAsFactors = FALSE)
  clinic_data$days[is.na(clinic_data$days)] <- 0
  clinic_data$status[is.na(clinic_data$status)] <- 0
  clinic_data$days <- as.numeric(clinic_data$days)
  clinic_data$days[is.na(clinic_data$days) | clinic_data$days < 0] <- 0
  return(clinic_data)
}

# 计算生存分析p值
get.empirical.pvalue <- function(pred_data, clinic_data, sub_name) {
  library(survival)
  
  # 构建生存分析的生存对象
  surv_object <- Surv(clinic_data$days, clinic_data$status)
  
  # 创建一个 Cox 比较模型
  surv_fit <- survdiff(surv_object ~ pred_data$label, data = clinic_data)
  
  # 获取p值
  pvalue <- surv_fit$pvalue
  
  # 【微小防爆补丁】：由于新版 R 环境有时不直接返回 pvalue 或 pvalue 极小等于 0
  # 这里加两行兜底，防止 -log10() 报错导致你一晚上的 Python 训练闪退。逻辑和原来一致。
  if (is.null(pvalue)) {
    pvalue <- 1 - pchisq(surv_fit$chisq, length(surv_fit$n) - 1)
  }
  if (pvalue == 0) { pvalue <- 1e-16 }
  
  neg_log_result <- -log10(pvalue)
  
  # 返回更详细的结果
  return(list(
    pvalue = pvalue,
    neg_log_pvalue = neg_log_result,
    chi_square = surv_fit$chisq,
    n = surv_fit$n
  ))
}


# 3. 临床特征富集分析

# 定义辅助函数
get.fixed.names <- function(patient.names, include.type=F) {
  # fix the TCGA names to only include the patient ids
  if (include.type) {
    return(gsub('-', '\\.', toupper(substring(patient.names, 1, 15))))
  } else {
    return(gsub('-', '\\.', toupper(substring(patient.names, 1, 12))))  
  }
}

get.clinical.params <- function(sub_name) {
  # 【唯一修改处】：把 D 盘路径换成了你的服务器路径
  clinical.data.path = paste0('/public/home/tanjingbo/CA-CAE-main/CACAE/Data/clinic/', sub_name, '.clinic')
  clinical.params = read.table(clinical.data.path,
                               sep=',', header=T, stringsAsFactors = F)
  rownames.with.duplicates = get.fixed.names(rownames(clinical.params))  
  clinical.params = clinical.params[!duplicated(rownames.with.duplicates),]
  rownames(clinical.params) = rownames.with.duplicates[!duplicated(rownames.with.duplicates)]
  return(clinical.params)
}

get.empirical.clinical <- function(clustering, clinical.values, is.chisq) {
  set.seed(42)
  valid <- !is.na(clinical.values) & !is.na(clustering)
  clinical.clean <- clinical.values[valid]
  clustering.clean <- clustering[valid]
  
  if (is.chisq) {
    # print('离散型')
    clustering.with.clinical = cbind(clustering.clean,clinical.clean)
    tbl = table(as.data.frame(clustering.with.clinical))
    test.res = chisq.test(tbl)
  } else {
    # print('数值型')
    test.res = kruskal.test(as.numeric(clinical.clean), clustering.clean)
  }
  orig.pvalue = test.res$p.value
  num.iter = 1000
  total.num.iters = 0
  total.num.extreme = 0
  should.continue = T
  while (should.continue) {
    # print('another iteration in empirical clinical') # 注释掉以防在服务器刷屏
    perm.pvalues = as.numeric(mclapply(1:num.iter, function(i) {
      cur.clustering = sample(clustering.clean)
      names(cur.clustering) = names(clustering)
      if (is.chisq) {
        clustering.with.clinical = cbind(cur.clustering, clinical.clean)
        tbl = table(as.data.frame(clustering.with.clinical))
        test.res = chisq.test(tbl)
      } else {
        test.res = kruskal.test(as.numeric(clinical.clean), cur.clustering)
      }
      cur.pvalue = test.res$p.value
      return(cur.pvalue)
    }, mc.cores=1))
    total.num.iters = total.num.iters + num.iter
    total.num.extreme = total.num.extreme + sum(perm.pvalues <= orig.pvalue)
    
    binom.ret = binom.test(total.num.extreme, total.num.iters)
    cur.pvalue = binom.ret$estimate
    cur.conf.int = binom.ret$conf.int
    
    sig.threshold = 0.05
    is.threshold.in.conf = cur.conf.int[1] < sig.threshold & cur.conf.int[2] > sig.threshold
    if (!is.threshold.in.conf | total.num.iters > 5e3) {
      should.continue = F
      
    }
  }
  return(cur.pvalue)
}

check.clinical.enrichment <- function(clustering, sub_name) {
  clinical.params = get.clinical.params(sub_name)  
  
  clinical.metadata = list(gender='DISCRETE', age_at_initial_pathologic_diagnosis='NUMERIC',
                           pathologic_M='DISCRETE', pathologic_N='DISCRETE', pathologic_T=
                             'DISCRETE', pathologic_stage='DISCRETE')
  
  pvalues = c()
  params.being.tested = c()
  for (clinical.param in names(clinical.metadata)) {#遍历每一个临床标签
    if (!(clinical.param %in% colnames(clinical.params))) {
      # print(paste0('WARNING: ', clinical.param, ' does not appear for subtype ', sub_name))
      next
    }
    #样本名一致筛选
    clinical.values = clinical.params[clinical.params$bcr_patient_barcode %in% 
                                        clustering$sample_name,clinical.param] #sample_names
    
    is.discrete.param = clinical.metadata[clinical.param] == 'DISCRETE'
    is.numeric.param = clinical.metadata[clinical.param] == 'NUMERIC'
    stopifnot(is.discrete.param | is.numeric.param)
    
    # skip parameter if many missing values
    if (is.numeric.param){
      numeric.entries = !is.na(as.numeric(clinical.values))
      if (2 * sum(numeric.entries) < length(clinical.values)) {
        # print(paste0('WARNING: skipping on ', clinical.param, ' for subtype ', sub_name))
        next
      }
    } else {
      not.na.entries = !is.na(clinical.values)
      should.skip = F
      if (2 * sum(not.na.entries) < length(clinical.values)) {
        should.skip = T
      } else if (length(table(clinical.values[not.na.entries])) == 1) {
        should.skip = T
      }
      if (should.skip) {
        # print(paste0('Skipping ', clinical.param, ' due to insufficient data or missing values'))
        next
      }
    }
    params.being.tested = c(params.being.tested, clinical.param)
    if (is.discrete.param) {
      pvalue = get.empirical.clinical(clustering$label[!is.na(clinical.values)], 
                                      clinical.values[!is.na(clinical.values)], T) #y_pred
    } else if (is.numeric.param) {
      pvalue = get.empirical.clinical(clustering$label[numeric.entries], 
                                      as.numeric(clinical.values[numeric.entries]), F)
    }
    pvalues = c(pvalues, pvalue)
    
  }
  names(pvalues) = params.being.tested
  return(pvalues)
}

analyze_clusters <- function(dcgcn_data, sub_name) {
  tryCatch({
    # 1. 强制解析数据格式 (防御 rpy2 传输造成的类型变异)
    dcgcn_data$label <- as.numeric(as.character(dcgcn_data$label))
    dcgcn_data$sample_name <- as.character(dcgcn_data$sample_name)
    
    # 2. 获取本地原始临床数据
    clinic_data <- get.clinical.data(sub_name)
    
    # 3. 强制主键对齐 (Inner Join)
    # [核心修复]: 侦测 TCGA ID 的真实位置
    raw_clinic_ids <- rownames(clinic_data)
    
    # 如果行名是纯数字 (例如 "1", "2")，说明真实的 ID 在第一列
    if (grepl("^[0-9]+$", raw_clinic_ids[1])) {
      raw_clinic_ids <- as.character(clinic_data[, 1])
    }
    
    # 模仿 Python 端的规范化操作：大写 -> 替换 . 为 - -> 截取前 12 位
    formatted_clinic_ids <- toupper(gsub("\\.", "-", raw_clinic_ids))
    formatted_clinic_ids <- substr(formatted_clinic_ids, 1, 12)
    
    # 获取交集
    common_ids <- intersect(formatted_clinic_ids, dcgcn_data$sample_name)
    
    if(length(common_ids) == 0) {
      stop(sprintf("TCGA ID 对齐失败。Python 传入样本: %s..., R 实际提取样本: %s...", 
                   head(dcgcn_data$sample_name, 1), head(formatted_clinic_ids, 1)))
    }
    
    # 4. 截断并生成绝对对齐的数据矩阵
    # 利用 match 获取 common_ids 在 formatted_clinic_ids 中的准确行索引
    match_idx_clinic <- match(common_ids, formatted_clinic_ids)
    aligned_clinic <- clinic_data[match_idx_clinic, , drop = FALSE]
    
    # 🚨 强制将对齐后的行名重置为标准 12 位 TCGA ID，保护你下游的旧函数不崩溃
    rownames(aligned_clinic) <- common_ids 
    
    # 对齐传入的聚类特征数据
    aligned_dcgcn <- dcgcn_data[match(common_ids, dcgcn_data$sample_name), , drop = FALSE]
    
    # 5. 组装原始函数所需的 pred_Y 格式
    pred_Y <- list(label = as.numeric(aligned_dcgcn$label))
    
    # 6. 计算 P 值
    pvalue_result <- get.empirical.pvalue(pred_Y, aligned_clinic, sub_name)
    
    # 7. 富集分析
    enrichment.value <- check.clinical.enrichment(aligned_dcgcn, sub_name)
    enrichment_count <- sum(enrichment.value < 0.05, na.rm = TRUE)
    
    return(list(
      neg_log_pvalue = pvalue_result$neg_log_pvalue,
      enrichment_count = enrichment_count
    ))
    
  }, error = function(e) {
    # 强制打印真实报错，解除静默失败 (Silent Failure)
    message(sprintf("❌ [R端计算崩溃] 癌症: %s, 报错详情: %s", sub_name, e$message))
    
    # 崩溃时返回 0，确保 Python 网格搜索能够 continue，不阻断程序运行
    return(list(
      neg_log_pvalue = 0,
      enrichment_count = 0
    ))
  })
}


