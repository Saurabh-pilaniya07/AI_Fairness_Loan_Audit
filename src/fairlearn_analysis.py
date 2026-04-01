from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

def compute_fairness_metrics(y_true, y_pred, sensitive_feature):
    
    dp_diff = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive_feature
    )
    
    eo_diff = equalized_odds_difference(
        y_true, y_pred, sensitive_features=sensitive_feature
    )
    
    return dp_diff, eo_diff


from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.ensemble import RandomForestClassifier

def mitigate_bias(X_train, y_train, sensitive_feature):
    
    base_model = RandomForestClassifier()
    
    mitigator = ExponentiatedGradient(
        base_model,
        constraints=DemographicParity()
    )
    
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_feature)
    
    return mitigator