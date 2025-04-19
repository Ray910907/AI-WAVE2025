"""
Preprocessing utilities for fraud detection pipeline.
Contains all feature engineering functions used by the training pipeline.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm


def preprocess_accts(df_accts):
    """Convert account-level fields to numeric features."""
    df = df_accts.copy()
    df['account_age'] = df['ACCT_OPEN_DT'].astype(float)
    df['CANCEL_NO_CONTACT'] = df['CANCEL_NO_CONTACT'].astype(int)
    df['IS_DIGITAL'] = df['IS_DIGITAL'].astype(int)
    return df


def preprocess_id(df_id):
    """Clean and impute ID-level demographics."""
    df = df_id.copy()
    for col in ['AUM_AMT', 'YEARLYINCOMELEVEL', 'CNTY_CD']:
        if col in df:
            df[col] = df[col].fillna(df[col].median())
    if 'DATE_OF_BIRTH' in df:
        df['AGE_GROUP'] = df['DATE_OF_BIRTH'].astype(int)
    return df


def preprocess_transactions(df_txn):
    """Feature-engineer raw transactions."""
    df = df_txn.copy()
    df['TX_DATE'] = df['TX_DATE'].astype(int)
    df['is_credit'] = (df['DRCR'] == 1).astype(int)
    df['is_debit'] = (df['DRCR'] == 2).astype(int)
    df['TX_HOUR'] = df['TX_TIME'].astype(int)
    df['is_weekend'] = df['DAY_OF_WEEK'].isin(['Saturday','Sunday']).astype(int)
    df['is_night'] = ((df['TX_HOUR'] >= 22) | (df['TX_HOUR'] <= 6)).astype(int)
    df['is_mobile'] = df['mb_check'].astype(int)
    df['is_ebanking'] = df['eb_check'].astype(int)
    df['is_digital_channel'] = ((df['is_mobile']==1) | (df['is_ebanking']==1)).astype(int)
    df['is_internal_transfer'] = (df['OWN_TRANS_ID'] != 'ID99999').astype(int)
    df['has_shared_ip'] = df['SAME_NUMBER_IP'].astype(int)
    df['has_shared_device'] = df['SAME_NUMBER_UUID'].astype(int)
    return df


def aggregate_transaction_features(df_txn, time_window=None):
    """Aggregate transactions at the account level."""
    df = df_txn.copy()
    if time_window:
        max_d = df['TX_DATE'].max()
        df = df[df['TX_DATE'] >= (max_d - time_window)]
    accounts = df['ACCT_NBR'].unique()
    rows = []
    for acct in tqdm(accounts, desc="Aggregating TX features"):
        sub = df[df['ACCT_NBR'] == acct]
        feat = {
            'ACCT_NBR': acct,
            'tx_count': len(sub),
            'avg_tx_amount': sub['TX_AMT'].mean(),
            'max_tx_amount': sub['TX_AMT'].max(),
            'sum_tx_amount': sub['TX_AMT'].sum(),
            'night_tx_ratio': sub['is_night'].mean(),
            'weekend_tx_ratio': sub['is_weekend'].mean(),
            'digital_tx_ratio': sub['is_digital_channel'].mean(),
            'shared_ip_ratio': sub['has_shared_ip'].mean(),
            'shared_device_ratio': sub['has_shared_device'].mean(),
            'avg_balance': sub['PB_BAL'].mean(),
            'last_balance': sub['PB_BAL'].iloc[-1] if len(sub) > 0 else 0
        }
        rows.append(feat)
    agg = pd.DataFrame(rows)
    agg['credit_debit_ratio'] = agg['tx_count'] / agg['tx_count'].replace(0, 1)
    if time_window:
        agg['tx_velocity'] = agg['tx_count'] / time_window
    return agg.fillna(0)


def calculate_account_behavior_changes(df_txn, windows=[7, 14, 30]):
    """Compute feature changes between time windows."""
    window_dfs = {}
    max_d = df_txn['TX_DATE'].max()
    for w in windows:
        dfw = df_txn[df_txn['TX_DATE'] >= (max_d - w)]
        dfw_agg = aggregate_transaction_features(dfw)
        dfw_agg.rename(
            columns={c: f"{c}_{w}d" for c in dfw_agg.columns if c != 'ACCT_NBR'},
            inplace=True
        )
        window_dfs[w] = dfw_agg
    base = window_dfs[windows[-1]]
    for w in windows[:-1]:
        base = base.merge(window_dfs[w], on='ACCT_NBR', how='left')
    for i in range(len(windows)-1):
        s, l = windows[i], windows[i+1]
        base[f'tx_count_change_{s}d_to_{l}d'] = (
            base[f'tx_count_{l}d'] - base[f'tx_count_{s}d']
        ) / base[f'tx_count_{l}d'].replace(0, 1)
    return base.fillna(0)


def extract_network_features(df_txn):
    """Derive network metrics based on counterparties."""
    pairs = df_txn[['ACCT_NBR', 'OWN_TRANS_ACCT']]
    pairs = pairs[pairs['OWN_TRANS_ACCT'] != 'ACCT31429']
    counts = pairs.groupby(['ACCT_NBR', 'OWN_TRANS_ACCT']).size().reset_index(name='txn')
    conn = counts.groupby('ACCT_NBR').size().reset_index(name='conn_count')
    maxc = counts.groupby('ACCT_NBR')['txn'].max().reset_index(name='max_txn')
    tot = counts.groupby('ACCT_NBR')['txn'].sum().reset_index(name='tot_txn')
    conc = maxc.merge(tot, on='ACCT_NBR')
    conc['tx_concentration'] = conc['max_txn'] / conc['tot_txn']
    net = conn.merge(conc[['ACCT_NBR', 'tx_concentration']], on='ACCT_NBR')
    return net


def prepare_model_features(df_merged, network_feats=None):
    """Combine merged df and network features; drop ID columns."""
    df = df_merged.copy()
    if network_feats is not None:
        df = df.merge(network_feats, on='ACCT_NBR', how='left')
    drop_cols = [c for c in ['ACCT_NBR', 'CUST_ID'] if c in df.columns]
    X = df.drop(columns=drop_cols).values
    feature_names = [c for c in df.columns if c not in drop_cols]
    return X, feature_names
