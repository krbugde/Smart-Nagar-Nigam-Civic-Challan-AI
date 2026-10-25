import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_data(data_dir="."):
    print("Loading datasets...")
    dtypes = {'ticket_issued_date': 'str', 'hearing_date': 'str'}
    parse_dates = ['ticket_issued_date', 'hearing_date']
    
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    address_path = os.path.join(data_dir, 'addresses.csv')
    latlons_path = os.path.join(data_dir, 'latlons.csv')
    
    df_train = pd.read_csv(train_path, encoding="ISO-8859-1", dtype=dtypes, parse_dates=parse_dates, low_memory=False)
    df_test = pd.read_csv(test_path, encoding="ISO-8859-1", dtype=dtypes, parse_dates=parse_dates, low_memory=False)
    df_address = pd.read_csv(address_path, encoding="ISO-8859-1")
    df_latlons = pd.read_csv(latlons_path, encoding="ISO-8859-1")
    
    # Merge address and lat/lon
    df_id_latlons = pd.merge(df_address, df_latlons, how='inner', on='address')
    df_id_latlons = df_id_latlons.drop(columns=['address'])
    
    # Fill missing lat/lon with most frequent / mode
    lat_mode = df_id_latlons['lat'].mode()[0]
    lon_mode = df_id_latlons['lon'].mode()[0]
    df_id_latlons['lat'] = df_id_latlons['lat'].fillna(lat_mode)
    df_id_latlons['lon'] = df_id_latlons['lon'].fillna(lon_mode)
    
    return df_train, df_test, df_id_latlons

def preprocess(df_train, df_test, df_id_latlons):
    print("Preprocessing data...")
    # Filter training data: keep only tickets where violator was found responsible
    df_train = df_train[df_train['compliance'].notnull()].copy()
    y_train = df_train['compliance'].astype(int)
    train_ticket_ids = df_train['ticket_id']
    test_ticket_ids = df_test['ticket_id']
    
    # Columns to remove (leakage, target-related, or high-null)
    leakage_cols = ['payment_amount', 'payment_date', 'payment_status', 'balance_due',
                    'collection_status', 'compliance_detail', 'compliance',
                    'violator_name', 'inspector_name', 'violation_zip_code',
                    'non_us_str_code', 'grafitti_status']
    
    address_cols = ['mailing_address_str_number', 'mailing_address_str_name', 
                    'city', 'state', 'zip_code', 'country',
                    'violation_street_number', 'violation_street_name']
    
    # Merge lat/lon
    df_train = pd.merge(df_train, df_id_latlons, how='left', on='ticket_id')
    df_test = pd.merge(df_test, df_id_latlons, how='left', on='ticket_id')
    
    # Feature engineering: date difference
    df_train['hearing_issued_date_diff'] = (df_train['hearing_date'] - df_train['ticket_issued_date']).dt.days
    df_test['hearing_issued_date_diff'] = (df_test['hearing_date'] - df_test['ticket_issued_date']).dt.days
    
    date_cols = ['hearing_date', 'ticket_issued_date']
    
    # Drop unused columns
    drop_from_train = [c for c in (leakage_cols + address_cols + date_cols) if c in df_train.columns]
    drop_from_test = [c for c in (leakage_cols + address_cols + date_cols) if c in df_test.columns]
    
    df_train = df_train.drop(columns=drop_from_train)
    df_test = df_test.drop(columns=drop_from_test)
    
    # Categorical columns to encode
    cat_cols = ['agency_name', 'violation_code', 'disposition']
    threshold = 100
    unknown_cat = '<unknown>'
    
    for col in cat_cols:
        if col in df_train.columns:
            counts = df_train[col].value_counts()
            frequent_cats = counts[counts > threshold].index.tolist()
            
            # Replace infrequent categories with '<unknown>'
            df_train[col] = df_train[col].apply(lambda x: x if x in frequent_cats else unknown_cat)
            df_test[col] = df_test[col].apply(lambda x: x if x in frequent_cats else unknown_cat)
    
    # One-hot encoding
    df_train = pd.get_dummies(df_train, columns=[c for c in cat_cols if c in df_train.columns], dtype=int)
    df_test = pd.get_dummies(df_test, columns=[c for c in cat_cols if c in df_test.columns], dtype=int)
    
    # Drop violation_description if present (high cardinality text)
    if 'violation_description' in df_train.columns:
        df_train = df_train.drop(columns=['violation_description'])
    if 'violation_description' in df_test.columns:
        df_test = df_test.drop(columns=['violation_description'])
        
    # Align train and test features
    features = [c for c in df_train.columns if c != 'ticket_id']
    for f in features:
        if f not in df_test.columns:
            df_test[f] = 0
            
    df_train_features = df_train[features].copy()
    df_test_features = df_test[features].copy()
    
    # Impute missing values with training mean
    means = df_train_features.mean()
    df_train_features = df_train_features.fillna(means)
    df_test_features = df_test_features.fillna(means)
    
    return df_train_features, y_train, df_test_features, test_ticket_ids, features

def evaluate_and_train(X_train, y_train, X_test, test_ids, features):
    print("\n--- Training Gradient Boosting Classifier ---")
    gbm = GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=30,
        min_samples_split=500,
        min_samples_leaf=50,
        max_depth=8,
        max_features='sqrt',
        subsample=0.8,
        random_state=10
    )
    
    print("Fitting model...")
    gbm.fit(X_train, y_train)
    
    train_preds = gbm.predict(X_train)
    train_probs = gbm.predict_proba(X_train)[:, 1]
    
    train_acc = accuracy_score(y_train, train_preds)
    train_auc = roc_auc_score(y_train, train_probs)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Train ROC-AUC : {train_auc:.4f}")
    
    print("Running 5-fold cross-validation for ROC-AUC...")
    cv_scores = cross_val_score(gbm, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"CV ROC-AUC Scores: {cv_scores}")
    print(f"Mean CV ROC-AUC  : {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    # Predict on test set
    print("\nGenerating test predictions...")
    test_probs = gbm.predict_proba(X_test)[:, 1]
    predictions_df = pd.DataFrame({
        'ticket_id': test_ids,
        'compliance': test_probs
    }).set_index('ticket_id')
    
    predictions_df.to_csv('predictions.csv')
    print(f"Saved {len(predictions_df)} test predictions to 'predictions.csv'")
    print(predictions_df.head(10))
    
    # Save Feature Importance Plot
    print("\nPlotting Feature Importance...")
    top_n = 10
    feat_imp = pd.Series(gbm.feature_importances_, index=features).sort_values(ascending=False)[:top_n]
    
    plt.figure(figsize=(10, 5))
    feat_imp.plot(kind='barh', color='steelblue')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance Score')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.close()
    print("Saved feature_importance.png")
    
    # Save ROC Curve Plot
    print("Plotting ROC Curve...")
    fpr, tpr, _ = roc_curve(y_train, train_probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {train_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300)
    plt.close()
    print("Saved roc_curve.png")
    
    # Compare with Logistic Regression
    print("\n--- Evaluating Logistic Regression ---")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    lr = LogisticRegression(C=40, max_iter=1000, random_state=10)
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_train_scaled)[:, 1]
    lr_auc = roc_auc_score(y_train, lr_probs)
    print(f"Logistic Regression Train ROC-AUC: {lr_auc:.4f}")
    
    return predictions_df

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))
    df_train, df_test, df_id_latlons = load_data(data_dir)
    X_train, y_train, X_test, test_ids, features = preprocess(df_train, df_test, df_id_latlons)
    evaluate_and_train(X_train, y_train, X_test, test_ids, features)
    print("\n[SUCCESS] Pipeline completed successfully!")
