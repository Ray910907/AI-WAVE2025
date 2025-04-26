import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
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
        label = self.label[idx]
        

        txn_records = self.txn_df[self.txn_df['ACCT_NBR'] == acct]
        txn_feat_cols = txn_records.columns.difference(['ACCT_NBR'])
        txn_seq = txn_records[txn_feat_cols].values.astype(np.float32)

        return torch.tensor(acc_feat), torch.tensor(txn_seq), torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    acc_feats, txn_seqs, labels = zip(*batch)
    acc_feats = torch.stack(acc_feats)
    labels = torch.stack(labels)

    txn_lens = [seq.shape[0] for seq in txn_seqs]
    max_len = max(txn_lens)

    padded_txns = torch.zeros(len(batch), max_len, txn_seqs[0].shape[1])
    mask = torch.ones(len(batch), max_len, dtype=torch.bool)

    for i, seq in enumerate(txn_seqs):
        padded_txns[i, :seq.shape[0], :] = seq
        mask[i, :seq.shape[0]] = False

    return acc_feats, padded_txns, labels, mask


