import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve, confusion_matrix

def train_and_export():
    print("Loading data for training...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    dtypes = {'ticket_issued_date': 'str', 'hearing_date': 'str'}
    parse_dates = ['ticket_issued_date', 'hearing_date']
    
    df_train = pd.read_csv(os.path.join(base_dir, 'train.csv'), encoding="ISO-8859-1", dtype=dtypes, parse_dates=parse_dates, low_memory=False)
    df_address = pd.read_csv(os.path.join(base_dir, 'addresses.csv'), encoding="ISO-8859-1")
    df_latlons = pd.read_csv(os.path.join(base_dir, 'latlons.csv'), encoding="ISO-8859-1")
    
    # Merge address and lat/lon
    df_id_latlons = pd.merge(df_address, df_latlons, how='inner', on='address').drop(columns=['address'])
    lat_mean = float(df_id_latlons['lat'].mean())
    lon_mean = float(df_id_latlons['lon'].mean())
    df_id_latlons['lat'] = df_id_latlons['lat'].fillna(lat_mean)
    df_id_latlons['lon'] = df_id_latlons['lon'].fillna(lon_mean)
    
    # Filter training data for responsible tickets
    df_train = df_train[df_train['compliance'].notnull()].copy()
    y_train = df_train['compliance'].astype(int)
    
    # Merge lat/lon
    df_train = pd.merge(df_train, df_id_latlons, how='left', on='ticket_id')
    df_train['lat'] = df_train['lat'].fillna(lat_mean)
    df_train['lon'] = df_train['lon'].fillna(lon_mean)
    
    # Feature engineering: date difference
    df_train['hearing_issued_date_diff'] = (df_train['hearing_date'] - df_train['ticket_issued_date']).dt.days
    date_diff_mean = float(df_train['hearing_issued_date_diff'].mean())
    df_train['hearing_issued_date_diff'] = df_train['hearing_issued_date_diff'].fillna(date_diff_mean)
    
    # Numerical features
    num_cols = ['fine_amount', 'admin_fee', 'state_fee', 'late_fee', 'discount_amount', 
                'clean_up_cost', 'judgment_amount', 'lat', 'lon', 'hearing_issued_date_diff']
    
    for c in num_cols:
        df_train[c] = df_train[c].fillna(float(df_train[c].mean()))
        
    num_means = {c: float(df_train[c].mean()) for c in num_cols}
    
    # Categorical features
    cat_cols = ['agency_name', 'violation_code', 'disposition']
    threshold = 100
    unknown_cat = '<unknown>'
    
    top_categories = {}
    for col in cat_cols:
        counts = df_train[col].value_counts()
        cats = counts[counts > threshold].index.tolist()
        top_categories[col] = cats
        df_train[col] = df_train[col].apply(lambda x: x if x in cats else unknown_cat)
        
    df_encoded = pd.get_dummies(df_train[cat_cols], dtype=int)
    feature_names_cat = df_encoded.columns.tolist()
    
    X_train = pd.concat([df_train[num_cols], df_encoded], axis=1)
    all_feature_names = X_train.columns.tolist()
    
    print(f"Total training samples: {len(X_train)}")
    print(f"Total features: {len(all_feature_names)}")
    
    # Train Gradient Boosting Classifier
    print("Training Gradient Boosting model...")
    gbm = GradientBoostingClassifier(
        learning_rate=0.08,
        n_estimators=45,
        min_samples_split=500,
        min_samples_leaf=50,
        max_depth=7,
        max_features='sqrt',
        subsample=0.85,
        random_state=42
    )
    gbm.fit(X_train, y_train)
    
    train_probs = gbm.predict_proba(X_train)[:, 1]
    train_preds = gbm.predict(X_train)
    train_auc = float(roc_auc_score(y_train, train_probs))
    train_acc = float(accuracy_score(y_train, train_preds))
    
    print(f"Train Accuracy: {train_acc:.4f}, ROC-AUC: {train_auc:.4f}")
    
    # Cross validation score
    print("Computing 5-fold CV score...")
    cv_scores = cross_val_score(gbm, X_train, y_train, cv=5, scoring='roc_auc')
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    print(f"5-Fold CV ROC-AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    # Feature importances
    feat_imp = {feat: float(imp) for feat, imp in zip(all_feature_names, gbm.feature_importances_)}
    sorted_feat_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_train, train_preds).tolist()
    
    # Save artifacts
    artifacts_dir = os.path.join(base_dir, 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    model_path = os.path.join(artifacts_dir, 'blight_model.joblib')
    joblib.dump(gbm, model_path)
    print(f"Saved model to {model_path}")
    
    metadata = {
        'num_cols': num_cols,
        'num_means': num_means,
        'cat_cols': cat_cols,
        'top_categories': top_categories,
        'unknown_cat': unknown_cat,
        'feature_names_cat': feature_names_cat,
        'all_feature_names': all_feature_names,
        'metrics': {
            'train_accuracy': train_acc,
            'train_auc': train_auc,
            'cv_roc_auc_mean': cv_mean,
            'cv_roc_auc_std': cv_std,
            'confusion_matrix': cm
        },
        'top_features': sorted_feat_imp[:20]
    }
    
    metadata_path = os.path.join(artifacts_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    # Save a small sample dataset for app demonstration / quick testing
    sample_df = df_train[['ticket_id'] + num_cols + cat_cols + ['compliance']].sample(n=2000, random_state=42)
    sample_df.to_csv(os.path.join(artifacts_dir, 'sample_tickets.csv'), index=False)
    print("Saved sample_tickets.csv for app exploration.")

if __name__ == '__main__':
    train_and_export()
