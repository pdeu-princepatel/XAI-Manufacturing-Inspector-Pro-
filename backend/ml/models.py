from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def train_xgboost(X, y):
    """
    Trains an XGBoost classifier and returns the model and metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, preds)),
        'f1': float(f1_score(y_test, preds)),
        'roc_auc': float(roc_auc_score(y_test, probs))
    }
    
    return model, metrics
