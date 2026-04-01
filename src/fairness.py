def demographic_parity(df_test):
    young = df_test[df_test['age_group']=='young']['y_pred'].mean()
    adult = df_test[df_test['age_group']=='adult']['y_pred'].mean()
    
    return abs(young - adult)


def equal_opportunity(df_test):
    young = df_test[(df_test['age_group']=='young') & (df_test['y_true']==1)]
    adult = df_test[(df_test['age_group']=='adult') & (df_test['y_true']==1)]
    
    return young['y_pred'].mean(), adult['y_pred'].mean()