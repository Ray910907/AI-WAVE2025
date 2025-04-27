import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from model import AccountAwareTransformer, train, evaluate
from others import *

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train_file_reader import train_file_reader

def main():
    # Load data
    train_path = './comp_data/Train/'
    train_reader = train_file_reader(train_path)
    acc_info, label, transac_info = train_reader.get_info()
    transac_info['Frequency'] = transac_info.groupby('ACCT_NBR')['TX_DATE'].diff().fillna(-30)
    dataset = FraudDataset(acc_info, transac_info, label)
    train_data, eval_data = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    #train_data = balance_dataset_with_smote(train_data)
    print("Train data size:", len(train_data))
    print("Eval data size:", len(eval_data))
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AccountAwareTransformer(acc_info.shape[1]-1, transac_info.shape[1]-1).to(device)
    Epochs = 100
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    class_weights = torch.tensor([1.0, 7.0]).to(device)
    #criterion = nn.CrossEntropyLoss()
    

    gamma = 1.5
    # Training
    save_path = './model/checkpoint5.pth'
    for epoch in range(Epochs):
        criterion = FocalLoss(gamma=gamma, weight=class_weights)
        loss, accurancy, f1 = train(model, train_dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{Epochs}, Train_Loss: {loss:.4f}, Accuracy: {accurancy:.4f}, F1: {f1:.4f}")
        # Evaluation
        loss, accuracy, f1 = evaluate(model, eval_dataloader, criterion, device)
        print(f"Epoch {epoch+1}/{Epochs}, Eval_Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_path)
        #gamma = gamma + 0.1 if gamma < 3 else 3
  
    # Save the model
    
    torch.save(model.state_dict(), save_path)








if __name__ == "__main__":
    main()

    



