import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, f1_score

def balance_data(data, label):
    """
    Balances the dataset by oversampling the minority class.

    Parameters:
        data (numpy.ndarray): The feature data.
        label (numpy.ndarray): The binary labels (0 or 1).

    Returns:
        balanced_data (numpy.ndarray): The balanced feature data.
        balanced_label (numpy.ndarray): The balanced labels.
    """
    # Separate the data into two classes
    data_class_0 = data[label == 0]
    data_class_1 = data[label == 1]

    # Determine the size of the majority class
    max_size = max(len(data_class_0), len(data_class_1))

    # Resample the minority class to match the majority class size
    data_class_0_resampled = resample(data_class_0, replace=True, n_samples=max_size, random_state=42)
    data_class_1_resampled = resample(data_class_1, replace=True, n_samples=max_size, random_state=42)

    # Combine the resampled data
    balanced_data = np.vstack((data_class_0_resampled, data_class_1_resampled))
    balanced_label = np.hstack((np.zeros(max_size), np.ones(max_size)))

    return balanced_data, balanced_label

def split_train_test(data, label, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.
    Parameters:
        data (pandas.DataFrame): The feature data.
        label (str): The name of the label column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The random seed for reproducibility.
    Returns:
        X_train (pandas.DataFrame): Training feature data.
        X_test (pandas.DataFrame): Testing feature data.
        y_train (pandas.Series): Training labels.
        y_test (pandas.Series): Testing labels.
    """
    
    columns = data.columns.tolist()
    # Perform the train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        data, label, test_size=test_size, random_state=random_state
    )
    X_train = X_train.drop(columns=['ACCT_NBR'])
    #to dataframe
    X_train = pd.DataFrame(X_train, columns=columns)
    X_test = pd.DataFrame(X_test, columns=columns)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)



    return X_train, X_test, y_train, y_test

def print_result(prediction, label):
    """
    Prints the classification report, confusion matrix, AUC score, and F1 score.
    Parameters:
        prediction (numpy.ndarray): The predicted labels.
        label (numpy.ndarray): The true labels.
    """
    
    
    print("Classification Report:")
    print(classification_report(label, prediction))
    print("Confusion Matrix:")
    print(confusion_matrix(label, prediction))
    print("AUC Score:", roc_auc_score(label, prediction))
    print("F1 Score:", f1_score(label, prediction))

def vote_for_prediction(predictions, data):
    """
    對於同樣CUST_ID的預測結果進行投票，選擇出現次數最多的預測結果
    """
    df = pd.DataFrame(data)
    df['prediction'] = predictions

    df['vote'] = df.groupby('ACCT_NBR')['prediction'].transform(lambda x: x.mode()[0] if not x.mode().empty else 0) 
    vote_result = df.groupby('ACCT_NBR')['vote'].first().reset_index()



    return vote_result['vote'].values
