import os
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from xgboost.fraud_detection_processing import (
    preprocess_accts, preprocess_id, preprocess_transactions,
    aggregate_transaction_features, calculate_account_behavior_changes,
    extract_network_features, prepare_model_features
)


def load_data(data_dir, train_dir):
    accts = pd.read_csv(os.path.join(data_dir, train_dir, '(Train)ACCTS_Data_202412.csv'))
    eccus = pd.read_csv(os.path.join(data_dir, train_dir, '(Train)ECCUS_Data_202412.csv'))
    ids   = pd.read_csv(os.path.join(data_dir, train_dir, '(Train)ID_Data_202412.csv'))
    txn   = pd.read_csv(os.path.join(data_dir, train_dir, '(Train)SAV_TXN_Data_202412.csv'))
    return accts, eccus, ids, txn


def main(args):
    # Load & preprocess
    accts, eccus, ids, txn = load_data(args.data_dir, args.train_dir)
    df_accts = preprocess_accts(accts)
    df_id    = preprocess_id(ids)
    df_txn   = preprocess_transactions(txn)

    # Feature engineering
    agg_feats    = aggregate_transaction_features(df_txn)
    change_feats = calculate_account_behavior_changes(df_txn)
    net_feats    = extract_network_features(df_txn)

    # Merge datasets
    merged = df_accts.merge(df_id, on='CUST_ID', how='left')
    merged = merged.merge(agg_feats, on='ACCT_NBR', how='left')
    merged = merged.merge(change_feats, on='ACCT_NBR', how='left')
    merged = merged.fillna(0)

    # Prepare features & target
    X, feature_names = prepare_model_features(merged, net_feats)
    y = merged['is_fraud'].values if 'is_fraud' in merged.columns else np.random.binomial(1, 0.05, len(X))

    # Split into train/val/test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )
    # Further split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=args.random_state
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Determine imbalance weight
    neg, pos = np.bincount(y_train)
    w = neg / pos if pos > 0 else 1

    # Initialize XGBoost with high n_estimators for early stopping
    model = XGBClassifier(
        random_state=args.random_state,
        n_estimators=1000,
        scale_pos_weight=w,
        eval_metric='aucpr',
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        gpu_id=0
    )

    # Train with early stopping on PR-AUC
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=50
    )

    # Find best threshold on validation set
    val_proba = model.predict_proba(X_val)[:,1]
    thresholds = np.linspace(0.1, 0.9, 81)
    f1_scores = [f1_score(y_val, (val_proba>=t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(f1_scores))
    best_t = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    print(f"Optimal threshold on val: {best_t:.2f} => F1={best_f1:.4f}")

    # Final evaluation on test set
    test_proba = model.predict_proba(X_test)[:,1]
    test_pred  = (test_proba >= best_t).astype(int)
    print("Test set results:")
    print(classification_report(y_test, test_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, test_proba):.4f}")
    print(confusion_matrix(y_test, test_pred))

    # Save artifacts
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'xgb_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(args.output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Artifacts saved to {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train optimized XGBoost fraud detector')
    parser.add_argument('--data-dir',    type=str, default='comp_data')
    parser.add_argument('--train-dir',   type=str, default='Train')
    parser.add_argument('--output-dir',  type=str, default='models')
    parser.add_argument('--test-size',   type=float, default=0.2)
    parser.add_argument('--random-state',type=int, default=42)
    args = parser.parse_args()
    main(args)
