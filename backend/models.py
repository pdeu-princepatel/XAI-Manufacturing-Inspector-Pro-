from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
import shap
import lime
import lime.lime_tabular
import numpy as np

# ========== REGRESSION MODELS ==========

def train_linear_regression(X, y):
    """
    Trains a basic Linear Regression model to predict continuous values and returns both the trained model and its performance metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    metrics = {
        'r2': float(r2_score(y_test, preds)),
        'mse': float(mean_squared_error(y_test, preds)),
        'mae': float(mean_absolute_error(y_test, preds))
    }
    
    return model, metrics, X_train, X_test

def train_random_forest_regressor(X, y):
    """
    Trains a Random Forest Regressor and returns the model and metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    metrics = {
        'r2': float(r2_score(y_test, preds)),
        'mse': float(mean_squared_error(y_test, preds)),
        'mae': float(mean_absolute_error(y_test, preds))
    }
    
    return model, metrics, X_train, X_test

def train_xgboost_regressor(X, y):
    """
    Trains an XGBoost Regressor and returns the model and metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(random_state=42, n_estimators=100, max_depth=6, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    metrics = {
        'r2': float(r2_score(y_test, preds)),
        'mse': float(mean_squared_error(y_test, preds)),
        'mae': float(mean_absolute_error(y_test, preds))
    }
    
    return model, metrics, X_train, X_test

# ========== CLASSIFICATION MODELS ==========

def train_logistic_regression(X, y):
    """
    Trains a Logistic Regression classifier and returns the model and metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, preds)),
        'f1': float(f1_score(y_test, preds)),
        'roc_auc': float(roc_auc_score(y_test, probs))
    }
    
    return model, metrics, X_train, X_test

def train_xgboost_classifier(X, y):
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
    
    return model, metrics, X_train, X_test

def train_decision_tree_classifier(X, y):
    """
    Trains a Decision Tree classifier and returns the model and metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, preds)),
        'f1': float(f1_score(y_test, preds)),
        'roc_auc': float(roc_auc_score(y_test, probs))
    }
    
    return model, metrics, X_train, X_test

# ========== XAI / EXPLAINABILITY ==========

def generate_shap_explanation(model, X_train, X_sample, model_type='tree'):
    """
    Calculates SHAP values to clear up the "black box" by explaining how each feature influenced the model's specific prediction.
    Args:
        model: The trained AI model generating the prediction
        X_train: Background training data used to establish a baseline
        X_sample: The specific sensor reading (single row) we want to understand
        model_type: Use 'tree' for decision trees or 'linear' for simpler linear models
    Returns:
        A dictionary containing the calculated SHAP impacts and the expected base value
    """
    try:
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_train)
        
        shap_values = explainer.shap_values(X_sample)
        
        # Extracting the target class values since SHAP can return them in different formats depending on the model
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, take positive class
        
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        return {
            'shap_values': shap_values.tolist(),
            'base_value': float(explainer.expected_value) if not isinstance(explainer.expected_value, np.ndarray) else float(explainer.expected_value[0]),
            'feature_names': list(X_sample.columns)
        }
    except Exception as e:
        print(f"SHAP explanation error: {e}")
        return None

def generate_lime_explanation(model, X_train, X_sample, mode='classification'):
    """
    Creates a local LIME explanation to show which sensor readings were most critical for this specific prediction.
    Args:
        model: The trained AI model to be explained
        X_train: The background training data
        X_sample: The specific sensor reading (single row) we are evaluating
        mode: Choose 'classification' (pass/fail) or 'regression' (continuous value)
    Returns:
        A dictionary detailing the feature importance scores calculated by LIME
    """
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=list(X_train.columns),
            mode=mode,
            random_state=42
        )
        
        if mode == 'classification':
            exp = explainer.explain_instance(X_sample.values[0], model.predict_proba, num_features=len(X_train.columns))
        else:
            exp = explainer.explain_instance(X_sample.values[0], model.predict, num_features=len(X_train.columns))
        
        return {
            'lime_values': dict(exp.as_list()),
            'score': float(exp.score) if hasattr(exp, 'score') else None
        }
    except Exception as e:
        print(f"LIME explanation error: {e}")
        return None
