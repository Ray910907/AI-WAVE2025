import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from file_reader import file_reader
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np



def train_random_forest(data,label):

    # Initialize the Random Forest model
    model = RandomForestClassifier(random_state=777)

    # Perform 5-fold cross-validation and store predictions
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    cv_scores = []
    all_predictions = []
    all_true_labels = []

    for train_index, test_index in skf.split(data, label):
        X_train, X_test = data.iloc[train_index.tolist()], data.iloc[test_index.tolist()]
        y_train, y_test = label.iloc[train_index.tolist()], label.iloc[test_index.tolist()]

        # Train the model
        model.fit(X_train, y_train)

        # Predict on the test set
        predictions = model.predict(X_test)
        all_predictions.extend(predictions)
        all_true_labels.extend(y_test)

        # Calculate accuracy for this fold
        accuracy = accuracy_score(y_test, predictions)
        cv_scores.append(accuracy)

    # Print cross-validation results
    print("Cross-Validation Scores:", cv_scores)
    print(f"Mean Accuracy: {np.mean(cv_scores):.2f}")
    print(f"Standard Deviation: {np.std(cv_scores):.2f}")

    # Save predictions and true labels for analysis
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    print("Confusion Matrix:")
    print(confusion_matrix(all_true_labels, all_predictions))
    print("Classification Report:")
    print(classification_report(all_true_labels, all_predictions))
    print(f"ROC AUC: {roc_auc_score(all_true_labels, all_predictions):.4f}")
    print(f"F1 Score: {f1_score(all_true_labels, all_predictions):.4f}")

    # Train the model on the entire dataset
    model.fit(data, label)

    return model

def main():
    pth = './comp_data/Train/'
    reader = file_reader(pth)
    acc_info, label, id2index = reader.read_merge_info()
    train_random_forest(acc_info, label)


if __name__ == "__main__":
    main()