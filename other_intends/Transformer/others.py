import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
import random
from torch.utils.data import Subset
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import numpy as np

class FraudDataset(Dataset):
    def __init__(self, acc_df, txn_df, label):
        self.acc_df = acc_df.set_index('ACCT_NBR')
        self.txn_df = txn_df
        self.label = label
        self.account_ids = self.acc_df.index.tolist()

    def __len__(self):
        return len(self.account_ids)

    def __getitem__(self, idx):
        acct = self.account_ids[idx]
        acc_feat_cols = self.acc_df.columns.difference(['ACCT_NBR'])
        acc_feat = self.acc_df.loc[acct, acc_feat_cols].values.astype(np.float32)
        acc_scalar = StandardScaler()
        acc_feat = acc_scalar.fit_transform(acc_feat.reshape(-1, 1)).flatten()
        label = self.label[idx]
        

        txn_records = self.txn_df[self.txn_df['ACCT_NBR'] == acct]
        txn_records = txn_records.sort_values(by=['TX_DATE'], ascending=False)
        txn_feat_cols = txn_records.columns.difference(['ACCT_NBR'])
        txn_seq = txn_records[txn_feat_cols].values.astype(np.float32)
        txn_scalar = StandardScaler()
        txn_seq = txn_scalar.fit_transform(txn_seq)
        

        return torch.tensor(acc_feat), torch.tensor(txn_seq), torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    acc_feats, txn_seqs, labels = zip(*batch)
    acc_feats = torch.stack(acc_feats)
    labels = torch.stack(labels)

    txn_lens = [seq.shape[0] for seq in txn_seqs]
    max_len = max(txn_lens)

    padded_txns = torch.zeros(len(batch), max_len, txn_seqs[0].shape[1])
    mask = torch.ones(len(batch), max_len, dtype=torch.bool)
    importance = torch.zeros(len(batch), max_len)

    for i, seq in enumerate(txn_seqs):
        padded_txns[i, :seq.shape[0], :] = seq
        mask[i, :seq.shape[0]] = False

        if seq.shape[1] > 3:
            importance[i, :seq.shape[0]] = torch.abs(seq[:, 3])  # 假設第0維是交易金額，重要性基於金額

    # normalize importance between 0 and 1 per batch
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-6)

    return acc_feats, padded_txns, labels, mask, importance

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none',label_smoothing=0.1)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()




def balance_dataset_with_smote(dataset):
    acc_feats = []
    labels = []
    txn_seqs = []
    for i in range(len(dataset)):
        acc_feat, txn_seq, label = dataset[i]
        acc_feats.append(acc_feat.numpy())
        txn_seqs.append(txn_seq)
        labels.append(label.item())

    acc_feats = np.array(acc_feats)
    labels = np.array(labels)

    smote = SMOTE()
    acc_feats_resampled, labels_resampled = smote.fit_resample(acc_feats, labels)

    # Map from original acc_feats to txn_seq
    acc_feat_to_txn_seq = {tuple(feat): txn for feat, txn in zip(acc_feats, txn_seqs)}

    resampled_dataset = []
    for acc_feat, label in zip(acc_feats_resampled, labels_resampled):
        acc_feat_tensor = torch.tensor(acc_feat, dtype=torch.float32)
        txn_seq = acc_feat_to_txn_seq.get(tuple(acc_feat), torch.zeros(1, txn_seqs[0].shape[1]))
        label_tensor = torch.tensor(label, dtype=torch.long)
        resampled_dataset.append((acc_feat_tensor, txn_seq, label_tensor))

    class SMOTEDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    return SMOTEDataset(resampled_dataset)