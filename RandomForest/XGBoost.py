import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train_file_reader import train_file_reader
from utils import *
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



def xgboost_by_account(data,label):

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=888)
    X_train, y_train = balance_data(X_train, y_train)

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'max_depth': 12,
        'eta': 0.3,
        'eval_metric': 'logloss'
    }
    num_round = 100
    bst = xgb.train(params, dtrain, num_round)
    preds = bst.predict(dtest)
    predictions = [1 if p > 0.5 else 0 for p in preds]
    print_result(predictions, y_test)

def xgboost_by_transaction(data,label):

    X_train, X_test, y_train, y_test = split_train_test(data, label)
    #X_train, y_train = balance_data(X_train, y_train)

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'max_depth': 12,
        'eta': 0.3,
        'eval_metric': 'logloss'
    }
    num_round = 100
    bst = xgb.train(params, dtrain, num_round)
    preds = bst.predict(dtest)
    predictions = [1 if p > 0.5 else 0 for p in preds]
    print_result(predictions, y_test)
    ground_truth_vote = vote_for_prediction(y_test, X_test)
    pred_vote = vote_for_prediction(predictions, X_test)
    print_result(pred_vote, ground_truth_vote)

reader = train_file_reader('./comp_data/Train/')
#optional
#data1, label1 = reader.get_transaction_merge_into_account()
#xgboost_by_account(data1, label1)
#optional
data2, label2 = reader.get_account_merge_into_transaction()
xgboost_by_transaction(data2, label2)




