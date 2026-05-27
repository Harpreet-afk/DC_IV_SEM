import os
import time
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, r2_score, mean_absolute_error,
                             mean_squared_error)

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: XGBoost not installed. Skipping XGBoost models.")


# Feature columns used for training
FEATURE_COLS = [
    'T', 'X_ZN', 'X_AL',
    'avg_atomic_radius', 'electronegativity_diff',
    'valence_electron_conc', 'atomic_size_mismatch',
    'mixing_entropy', 'homologous_temperature'
]


def consolidate_phase_labels(df):
    df = df.copy()
    
    # Sort components in multi-phase labels
    def sort_phase(p):
        if '+' in p:
            return '+'.join(sorted(p.split('+')))
        return p
        
    df['Phase'] = df['Phase'].apply(sort_phase)
    
    # Check counts
    counts = df['Phase'].value_counts()
    
    # Find rare phases
    rare_phases = counts[counts < 5].index.tolist()
    
    if rare_phases:
        print(f"Consolidating rare phases into 'Multi-phase': {rare_phases}")
        df.loc[df['Phase'].isin(rare_phases), 'Phase'] = 'Multi-phase'
        
        final_counts = df['Phase'].value_counts()
        if 'Multi-phase' in final_counts and final_counts['Multi-phase'] < 2:
            rare_rows = df[df['Phase'] == 'Multi-phase']
            df = pd.concat([df, rare_rows], ignore_index=True)
            print("Duplicated rare 'Multi-phase' samples to allow stratified split.")
            
    return df


def prepare_data(df, target_col='Phase', test_size=0.2, random_state=42):
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    if target_col == 'Phase':
        df = consolidate_phase_labels(df)

    X = df[feature_cols].values
    y = df[target_col].values

    scaler = StandardScaler()
    label_encoder = None

    if target_col == 'Phase':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if target_col == 'Phase' else None
    )

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder, feature_cols



def get_classification_models():
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
        'MLP Neural Network': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                                            random_state=42, early_stopping=True),
    }
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                                          random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    return models


def get_regression_models():
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
        'MLP Neural Network': MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                                           random_state=42, early_stopping=True),
    }
    if HAS_XGBOOST:
        models['XGBoost'] = XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42)
    return models


def train_classification(df, models_dir='models'):
    print("\n" + "=" * 60)
    print("  CLASSIFICATION: Predicting Stable Phase")
    print("=" * 60)

    X_train, X_test, y_train, y_test, scaler, le, feat_cols = prepare_data(df, 'Phase')
    models = get_classification_models()
    results = {}

    os.makedirs(models_dir, exist_ok=True)

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - t0

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            'accuracy': round(acc, 4),
            'f1_score': round(f1, 4),
            'train_time': round(train_time, 3),
            'pred_time': round(pred_time, 5),
            'pred_time_per_sample': round(pred_time / len(y_test) * 1000, 5),
            'confusion_matrix': cm.tolist(),
        }

        print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f}")
        print(f"  Train time: {train_time:.3f}s | Pred time: {pred_time:.5f}s ({pred_time/len(y_test)*1e6:.1f} us/sample)")

        # Save model
        model_path = os.path.join(models_dir, f"clf_{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)

    # Save scaler and label encoder
    joblib.dump(scaler, os.path.join(models_dir, 'clf_scaler.pkl'))
    joblib.dump(le, os.path.join(models_dir, 'clf_label_encoder.pkl'))
    joblib.dump(feat_cols, os.path.join(models_dir, 'clf_feature_cols.pkl'))

    return results, models, scaler, le


def train_regression(df, target='Gibbs_energy', models_dir='models'):
    print(f"\n{'=' * 60}")
    print(f"  REGRESSION: Predicting {target}")
    print("=" * 60)

    X_train, X_test, y_train, y_test, scaler, _, feat_cols = prepare_data(df, target)
    models = get_regression_models()
    results = {}

    os.makedirs(models_dir, exist_ok=True)

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - t0

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results[name] = {
            'r2_score': round(r2, 4),
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'train_time': round(train_time, 3),
            'pred_time': round(pred_time, 5),
        }

        print(f"  R²: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")
        print(f"  Train time: {train_time:.3f}s | Pred time: {pred_time:.5f}s")

        tag = target.lower().replace(' ', '_')
        model_path = os.path.join(models_dir, f"reg_{tag}_{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)

    joblib.dump(scaler, os.path.join(models_dir, f'reg_{target.lower()}_scaler.pkl'))

    return results, models, scaler


def predict_phase(T, x_zn, models_dir='models'):
    from src.feature_engineering import add_engineered_features

    df_input = pd.DataFrame({'T': [T], 'X_ZN': [x_zn], 'X_AL': [1.0 - x_zn]})
    df_input = add_engineered_features(df_input)

    feat_cols = joblib.load(os.path.join(models_dir, 'clf_feature_cols.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'clf_scaler.pkl'))
    le = joblib.load(os.path.join(models_dir, 'clf_label_encoder.pkl'))

    X = scaler.transform(df_input[feat_cols].values)
    predictions = {}

    for fname in os.listdir(models_dir):
        if fname.startswith('clf_') and fname.endswith('.pkl') and 'scaler' not in fname and 'label' not in fname and 'feature' not in fname:
            model = joblib.load(os.path.join(models_dir, fname))
            model_name = fname.replace('clf_', '').replace('.pkl', '').replace('_', ' ').title()
            pred_encoded = model.predict(X)[0]
            pred_phase = le.inverse_transform([pred_encoded])[0]
            predictions[model_name] = pred_phase

    return predictions


def get_ml_predictions_grid(df, model, scaler, le, feat_cols):
    X = scaler.transform(df[feat_cols].values)
    y_pred_encoded = model.predict(X)
    y_pred_phases = le.inverse_transform(y_pred_encoded)
    return y_pred_phases


def save_metrics(clf_results, reg_gibbs_results, reg_frac_results, output_path):
    metrics = {
        'classification': clf_results,
        'regression_gibbs_energy': reg_gibbs_results,
        'regression_phase_fraction': reg_frac_results,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {output_path}")
