import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_file_reader import train_file_reader
from utils import *




def random_forest_by_account(data, label):
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=888)
    #X_train = X_train.drop(columns=['ACCT_NBR'], inplace=True)
    #X_train, y_train = balance_data(X_train, y_train)


    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print_result(y_pred, y_test)



def random_forest_by_transaction(data, label):
    X_train, X_test, y_train, y_test = split_train_test(data, label)
    #X_train = X_train.drop(columns=['ACCT_NBR'], inplace=True)
    #X_train, y_train = balance_data(X_train, y_train)


    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print_result(y_pred, y_test)
    ground_truth_vote = vote_for_prediction(y_test, X_test)
    pred_vote = vote_for_prediction(y_pred, X_test)
    print_result(pred_vote, ground_truth_vote)




path = './comp_data/Train/'
reader = train_file_reader(path)

#optional
data1, label1 = reader.get_transaction_merge_into_account()
random_forest_by_account(data1, label1)

#optional
#data2, label2 = reader.get_account_merge_into_transaction()
#random_forest_by_transaction(data2, label2)
