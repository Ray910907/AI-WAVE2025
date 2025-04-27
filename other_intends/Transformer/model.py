import torch
from sklearn.metrics import f1_score
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

class AccountAwareTransformer(nn.Module):
    def __init__(self, account_feat_dim, txn_feat_dim, model_dim=128, num_heads=4, num_layers=2, num_classes=2, dropout=0.1):
        super(AccountAwareTransformer, self).__init__()
        self.acc_embedding = nn.Linear(account_feat_dim, model_dim)
        self.txn_embedding = nn.Linear(txn_feat_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, acc_feat, txn_seq, mask=None, importance=None):
        acc_token = self.acc_embedding(acc_feat).unsqueeze(1)
        txn_tokens = self.txn_embedding(txn_seq)
        x = torch.cat([acc_token, txn_tokens], dim=1)

        if mask is not None:
            account_mask = torch.zeros((mask.shape[0], 1), dtype=torch.bool, device=mask.device)
            mask = torch.cat([account_mask, mask], dim=1)

        if importance is not None:
            account_importance = torch.zeros((importance.shape[0], 1), device=importance.device)
            importance = torch.cat([account_importance, importance], dim=1)
            # normalize importance
            importance = importance.unsqueeze(2)
            x = x * (1 + importance)

        x = self.transformer(x, src_key_padding_mask=mask)
        cls_token = x[:, 0, :]
        return self.classifier(cls_token)                   
    
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    for acc_feat, txn_seq, labels, mask, importance in tqdm(train_loader, desc="Training", leave=False,ncols=100):
        acc_feat, txn_seq, labels = acc_feat.to(device), txn_seq.to(device), labels.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        outputs = model(acc_feat, txn_seq, mask, importance)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_labels = [1 if i> 0.5 else 0 for i in all_labels]
        all_predictions.extend(predicted.cpu().numpy())
        all_predictions = [1 if i> 0.5 else 0 for i in all_predictions]

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))  
    f1 = f1_score(all_labels, all_predictions)
    
    return total_loss / len(train_loader), correct / total, f1

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for acc_feat, txn_seq, labels, mask,importance in val_loader:
            acc_feat, txn_seq, labels = acc_feat.to(device), txn_seq.to(device), labels.to(device)
            mask = mask.to(device)

            outputs = model(acc_feat, txn_seq, mask, importance)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_labels = [1 if i> 0.5 else 0 for i in all_labels]
            all_predictions.extend(predicted.cpu().numpy())
            all_predictions = [1 if i> 0.5 else 0 for i in all_predictions]

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions)) 
    f1 = f1_score(all_labels, all_predictions)
    return total_loss / len(val_loader), correct / total, f1
