import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train_file_reader import train_file_reader
from test_file_reader import test_file_reader
from utils import *
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
import seaborn as sns

threshold = 0.6

def visualize_results(X_test, y_test, y_pred_proba, feature_importance, feature_names):
        """
        Create visualizations for model evaluation
        
        Parameters:
        - X_test, y_test: Test data
        - y_pred_proba: Prediction probabilities
        """
        print("Creating visualizations...")
        
        # 1. Feature Importance Plot
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(15)
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title('Top 15 Feature Importance (Gain)')
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png')
        
        # 2. ROC Curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('plots/roc_curve.png')
        
        # 3. Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall Curve: AP={avg_precision:.3f}')
        plt.savefig('plots/precision_recall_curve.png')
        
        # 5. Confusion Matrix
        y_pred = (y_pred_proba > threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('plots/confusion_matrix.png')
        
        print("Visualizations saved to 'plots' directory")
    
def generate_shap_plots(X_test, model, feature_names):
    """
    Generate SHAP value plots for model interpretation
    
    Parameters:
    - X_test: Test data features
    """
    try:
        import shap
        print("Generating SHAP plots for model interpretation...")
        
        # Create explainer
        explainer = shap.TreeExplainer(model)
        
        # Sample data for SHAP analysis (use subset if dataset is large)
        if X_test.shape[0] > 500:
            shap_sample = X_test[:500]
        else:
            shap_sample = X_test
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(shap_sample)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, shap_sample, feature_names=feature_names, show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig('plots/shap_summary.png')
        plt.close()
        
        # Dependence plots for top features
        top_features_idx = np.argsort(np.abs(shap_values).mean(0))[-5:]
        for i in top_features_idx:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(i, shap_values, shap_sample, feature_names=feature_names, show=False)
            plt.title(f'SHAP Dependence Plot: {feature_names[i]}')
            plt.tight_layout()
            plt.savefig(f'plots/shap_dependence_{feature_names[i]}.png')
            plt.close()
        
        print("SHAP plots saved to 'plots' directory")
        
    except ImportError:
        print("SHAP package not installed. Install with: pip install shap")

def encode_categorical(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

import pandas as pd

def predict_and_save(model, test_data_path):
    
    reader = test_file_reader(test_data_path)

    test_data = reader.get_transaction_merge_into_account()

    acct_nbr = test_data['ACCT_NBR'].values if 'ACCT_NBR' in test_data.columns else None

    if acct_nbr is not None:
        test_data = test_data.drop(columns=['ACCT_NBR'])

    preds = model.predict(test_data)


    predictions = [1 if p > threshold else 0 for p in preds]
    print(f'Num of fraud: {sum(predictions)}')

    reader.write_answer(pd.Series(predictions))


def lightgbm_by_account(data, label):
    with open('columns_1.txt', 'w') as f:
        for column in data.columns:
            f.write(f"{column}\n")
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    #X_train, y_train = data, label
    X_train, y_train = balance_data(X_train, y_train)

    # Create Dataset for LightGBM
    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    # Set LightGBM parameters
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 52,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 2,
        'verbose': -1,
        'scale_pos_weight': scale_pos_weight
    }
    num_round = 1000
    bst = lgb.train(params, dtrain, num_boost_round=num_round)

    preds = bst.predict(X_test)

    importance = bst.feature_importance(importance_type='gain')

    feature_names = X_test.columns.tolist()
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    visualize_results(X_test,y_test,preds,feature_importance,feature_names)
    generate_shap_plots(X_test,bst,feature_names)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(preds)), preds, c=y_test, cmap='viridis', alpha=0.5)
    plt.colorbar(label='True Label')
    plt.xlabel('Sample Index')
    plt.ylabel('Prediction Probability')
    plt.title('Prediction Probabilities with True Labels')
    plt.savefig('./plots/light.png')

    predictions = [1 if p > threshold else 0 for p in preds]
    print_result(predictions, y_test)

    return bst


def main():
    reader = train_file_reader('./comp_data/Train/')

    #optional

    data1, label1 = reader.get_transaction_merge_into_account()
    
    bst = lightgbm_by_account(data1, label1) 

    predict_and_save(bst, './comp_data/Test/')

if __name__ == "__main__":
    main()
