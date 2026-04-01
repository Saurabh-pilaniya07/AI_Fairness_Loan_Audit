def preprocess(df):
    df['target'] = df['credit_risk'].apply(lambda x: 1 if x == 1 else 0)  
    print("Target null values:", df['target'].isnull().sum())
      
    df['age_group'] = df['age'].apply(lambda x: 'young' if x < 25 else 'adult')
    
    features = ['age', 'amount', 'duration']
    
    X = df[features]
    y = df['target']
    
    return X, y, df