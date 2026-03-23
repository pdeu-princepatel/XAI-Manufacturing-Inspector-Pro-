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
    Trains a Linear Regression model and returns the model and metrics.
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
    Generates SHAP values for a given model and sample.
    Args:
        model: Trained model
        X_train: Training data for background
        X_sample: Sample to explain (single row as DataFrame)
        model_type: 'tree' for tree-based models, 'linear' for linear models
    Returns:
        dict with SHAP values and base value
    """
    try:
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_train)
        
        shap_values = explainer.shap_values(X_sample)
        
        # Handle both 1D and 2D SHAP values
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
    Generates LIME explanation for a given model and sample.
    Args:
        model: Trained model
        X_train: Training data for background
        X_sample: Sample to explain (single row as DataFrame)
        mode: 'classification' or 'regression'
    Returns:
        dict with feature importance from LIME
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
