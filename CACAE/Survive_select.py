def survive_select(survive_data, data, p_thresh, lasso_alpha=0.05):

    from lifelines import CoxPHFitter
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler

    time_col = survive_data['days'].values
    event_col = survive_data['status'].values

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    lasso = Lasso(alpha=lasso_alpha)
    lasso.fit(scaled_data, time_col)

    selected_features_indices = [i for i, coef in enumerate(lasso.coef_) if coef != 0]

    selected_features = []

    # for i in selected_features_indices:
    #     cpf = CoxPHFitter()
    #     temp_df = survive_data.copy()
    #     temp_df['feature'] = data.iloc[:, i].values
    #     cpf.fit(temp_df, 'days', 'status')

    #     if cpf.summary['p'].values[0] <= p_thresh:
    #         selected_features.append(i)
    
    for i in selected_features_indices:
        cpf = CoxPHFitter()
        
        # ====== 修改这里：只提取我们需要的三列，屏蔽掉 FEMALE 等字符串列 ======
        import pandas as pd
        temp_df = pd.DataFrame({
            'days': time_col,         # 生存天数
            'status': event_col,      # 生死状态
            'feature': data.iloc[:, i].values  # 当前测试的组学特征
        })
        # ======================================================================
        
        cpf.fit(temp_df, 'days', 'status')

        if cpf.summary['p'].values[0] <= p_thresh:
            selected_features.append(i)

    selected_data = data.iloc[:, selected_features]

    selected_data.index = data.index

    return selected_data
