#!/usr/bin/env python3
import sys, json, os
import joblib
from dataset import build_training_sets
from models import (
    train_linear_regression,
    train_random_forest_regressor,
    train_xgboost_regressor,
    train_logistic_regression,
    train_xgboost_classifier,
    train_decision_tree_classifier,
    generate_shap_explanation,
    generate_lime_explanation
)

# Ensure models directory exists
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

all_metrics = {}

try:
    print("=" * 60)
    print("TRAINING REGRESSION MODELS")
    print("=" * 60)
    
    # Load regression dataset
    X_reg, y_reg = build_training_sets(target_type='regression')
    
    # 1. Linear Regression
    print("\n[1/6] Training Linear Regression...")
    lr_model, lr_metrics, lr_X_train, lr_X_test = train_linear_regression(X_reg, y_reg)
    joblib.dump(lr_model, os.path.join(MODELS_DIR, 'linear_regression.joblib'))
    all_metrics['linear_regression'] = lr_metrics
    print(f"  R²: {lr_metrics['r2']:.4f}, MAE: {lr_metrics['mae']:.4f}")
    
    # 2. Random Forest Regressor
    print("\n[2/6] Training Random Forest Regressor...")
    rf_model, rf_metrics, rf_X_train, rf_X_test = train_random_forest_regressor(X_reg, y_reg)
    joblib.dump(rf_model, os.path.join(MODELS_DIR, 'random_forest_regressor.joblib'))
    all_metrics['random_forest_regressor'] = rf_metrics
    print(f"  R²: {rf_metrics['r2']:.4f}, MAE: {rf_metrics['mae']:.4f}")
    
    # 3. XGBoost Regressor
    print("\n[3/6] Training XGBoost Regressor...")
    xgb_reg_model, xgb_reg_metrics, xgb_reg_X_train, xgb_reg_X_test = train_xgboost_regressor(X_reg, y_reg)
    joblib.dump(xgb_reg_model, os.path.join(MODELS_DIR, 'xgboost_regressor.joblib'))
    all_metrics['xgboost_regressor'] = xgb_reg_metrics
    print(f"  R²: {xgb_reg_metrics['r2']:.4f}, MAE: {xgb_reg_metrics['mae']:.4f}")
    
    print("\n" + "=" * 60)
    print("TRAINING CLASSIFICATION MODELS")
    print("=" * 60)
    
    # Load classification dataset
    X_clf, y_clf = build_training_sets(target_type='classification')
    
    # 4. Logistic Regression
    print("\n[4/6] Training Logistic Regression...")
    log_model, log_metrics, log_X_train, log_X_test = train_logistic_regression(X_clf, y_clf)
    joblib.dump(log_model, os.path.join(MODELS_DIR, 'logistic_regression.joblib'))
    all_metrics['logistic_regression'] = log_metrics
    print(f"  Accuracy: {log_metrics['accuracy']:.4f}, F1: {log_metrics['f1']:.4f}")
    
    # 5. XGBoost Classifier
    print("\n[5/6] Training XGBoost Classifier...")
    xgb_clf_model, xgb_clf_metrics, xgb_clf_X_train, xgb_clf_X_test = train_xgboost_classifier(X_clf, y_clf)
    joblib.dump(xgb_clf_model, os.path.join(MODELS_DIR, 'xgboost_classifier.joblib'))
    all_metrics['xgboost_classifier'] = xgb_clf_metrics
    print(f"  Accuracy: {xgb_clf_metrics['accuracy']:.4f}, F1: {xgb_clf_metrics['f1']:.4f}")
    
    # 6. Decision Tree Classifier
    print("\n[6/6] Training Decision Tree Classifier...")
    dt_model, dt_metrics, dt_X_train, dt_X_test = train_decision_tree_classifier(X_clf, y_clf)
    joblib.dump(dt_model, os.path.join(MODELS_DIR, 'decision_tree_classifier.joblib'))
    all_metrics['decision_tree_classifier'] = dt_metrics
    print(f"  Accuracy: {dt_metrics['accuracy']:.4f}, F1: {dt_metrics['f1']:.4f}")
    
    print("\n" + "=" * 60)
    print("GENERATING XAI EXPLANATIONS (Sample)")
    print("=" * 60)
    
    # Generate sample explanations for XGBoost Classifier
    sample_row = xgb_clf_X_test.iloc[[0]]
    
    print("\n[SHAP] Generating explanation for XGBoost Classifier...")
    shap_result = generate_shap_explanation(xgb_clf_model, xgb_clf_X_train, sample_row, model_type='tree')
    if shap_result:
        print(f"  Base value: {shap_result['base_value']:.4f}")
        print(f"  Top 3 features by SHAP value:")
        feature_importance = sorted(zip(shap_result['feature_names'], shap_result['shap_values']), 
                                    key=lambda x: abs(x[1]), reverse=True)
        for feat, val in feature_importance[:3]:
            print(f"    {feat}: {val:.4f}")
    
    print("\n[LIME] Generating explanation for XGBoost Classifier...")
    lime_result = generate_lime_explanation(xgb_clf_model, xgb_clf_X_train, sample_row, mode='classification')
    if lime_result:
        print(f"  Top 3 features by LIME importance:")
        for i, (feat, val) in enumerate(list(lime_result['lime_values'].items())[:3]):
            print(f"    {feat}: {val:.4f}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # Output final metrics as JSON for Node.js backend
    print(json.dumps(all_metrics))
    
except Exception as e:
    # Print error to stderr so Node.js catches it
    print(f"Error: {str(e)}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
