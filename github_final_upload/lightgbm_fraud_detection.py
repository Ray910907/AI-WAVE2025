import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from train_file_reader import train_file_reader
import time
warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(42)

class LightGBMFraudDetector:
    def __init__(self, data_path='./comp_data/Train/', use_gpu=True):
        """
        Initialize the fraud detector with the file reader
        
        Parameters:
        - data_path: Path to data directory
        - use_gpu: Whether to use GPU acceleration
        """
        self.data_path = data_path
        self.use_gpu = use_gpu
        self.reader = train_file_reader(data_path)
        self.model = None
        self.best_threshold = 0.5
        self.feature_importance = None
        
        # Create directories for outputs
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        
        print(f"Initialized LightGBM Fraud Detector with data from {data_path}")
        print(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    
    def prepare_data(self, strategy='account_based'):
        """
        Prepare data based on strategy
        
        Parameters:
        - strategy: 'account_based' or 'transaction_based'
        
        Returns:
        - X_train, X_test, y_train, y_test, feature_names
        """
        print(f"Preparing data using {strategy} strategy...")
        
        if strategy == 'account_based':
            # Use transaction features merged into account info
            X, y = self.reader.get_transaction_merge_into_account()
        elif strategy == 'transaction_based':
            # Use account features merged into transaction info
            # This approach is more complex as it has multiple transactions per account
            # For simplicity, we'll focus on account-based approach
            raise NotImplementedError("Transaction-based strategy not yet implemented")
        else:
            raise ValueError("Invalid strategy. Choose 'account_based' or 'transaction_based'")
        
        # Convert to numpy arrays
        X_data = X.values
        y_data = y.values
        
        # Save feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scaler = scaler  # Save for inference
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Testing set: {X_test_scaled.shape}")
        print(f"Positive samples (fraud) in training: {np.sum(y_train)}")
        print(f"Positive samples (fraud) in testing: {np.sum(y_test)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, self.feature_names
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """
        Train LightGBM model
        
        Parameters:
        - X_train, y_train: Training data
        - X_test, y_test: Testing data
        
        Returns:
        - Trained model
        """
        print("Training LightGBM model...")
        
        # Calculate class weights for imbalanced dataset
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        # LightGBM dataset format
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Parameters with GPU support if enabled
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 52,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbose': -1,
            'scale_pos_weight': scale_pos_weight
        }
        
        # Add GPU parameters if enabled
        if self.use_gpu:
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0
        
        # Track training time
        start_time = time.time()
        
        # Train model
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data]
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save the model
        model.save_model('models/fraud_detection_model.txt')
        print("Model saved to models/fraud_detection_model.txt")
        
        self.model = model
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate model performance
        
        Parameters:
        - model: Trained LightGBM model
        - X_test, y_test: Test data
        
        Returns:
        - Dictionary with evaluation metrics
        """
        print("Evaluating model performance...")
        
        # Get predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Optimize threshold for F1 score
        best_threshold, y_pred_optimized = self.optimize_threshold(y_test, y_pred_proba)
        self.best_threshold = best_threshold
        
        # Calculate feature importance
        importance = model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
        
        self.feature_importance = feature_importance
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
        
        # Save metrics
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'best_threshold': best_threshold,
            'optimized_precision': precision_score(y_test, y_pred_optimized),
            'optimized_recall': recall_score(y_test, y_pred_optimized),
            'optimized_f1': f1_score(y_test, y_pred_optimized)
        }
        
        return metrics, y_pred_proba
    
    def optimize_threshold(self, y_true, y_pred_proba):
        """
        Find optimal threshold to maximize F1 score
        
        Parameters:
        - y_true: True labels
        - y_pred_proba: Prediction probabilities
        
        Returns:
        - best_threshold: Optimal threshold
        - y_pred_optimized: Predictions using optimal threshold
        """
        print("Optimizing threshold for F1 score...")
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            f1_scores.append(f1)
        
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        print(f"Optimal threshold: {best_threshold:.2f} with F1 score: {best_f1:.4f}")
        
        # Make predictions with optimized threshold
        y_pred_optimized = (y_pred_proba > best_threshold).astype(int)
        
        # Calculate metrics with optimized threshold
        precision = precision_score(y_true, y_pred_optimized)
        recall = recall_score(y_true, y_pred_optimized)
        
        print(f"Optimized Precision: {precision:.4f}")
        print(f"Optimized Recall: {recall:.4f}")
        
        return best_threshold, y_pred_optimized
    
    def cross_validate(self, X, y, n_splits=5):
        """
        Perform cross-validation
        
        Parameters:
        - X, y: Full dataset
        - n_splits: Number of CV folds
        
        Returns:
        - Average metrics
        """
        print(f"Performing {n_splits}-fold cross-validation...")
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Metrics for each fold
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        # Basic parameters without any callbacks or early stopping
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 52,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbose': -1
        }
        
        # Add GPU parameters if enabled
        if self.use_gpu:
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0
        
        # Perform cross-validation
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"Fold {fold+1}/{n_splits}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Calculate class weights for imbalanced dataset
            pos_count = np.sum(y_train == 1)
            neg_count = np.sum(y_train == 0)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            params['scale_pos_weight'] = scale_pos_weight
            
            # LightGBM dataset format
            train_data = lgb.Dataset(X_train, label=y_train)
            
            # The simplest possible train call with no callbacks or early stopping
            model = lgb.train(
                params,
                train_data,
                num_boost_round=100  # Fixed number of rounds, no early stopping
            )
            
            # Predictions
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Metrics
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            
            print(f"Fold {fold+1} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Average metrics
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_f1 = np.mean(f1_scores)
        
        print(f"\nAverage Cross-Validation Metrics:")
        print(f"Precision: {avg_precision:.4f}")
        print(f"Recall: {avg_recall:.4f}")
        print(f"F1 Score: {avg_f1:.4f}")
        
        return avg_precision, avg_recall, avg_f1
    
    def visualize_results(self, X_test, y_test, y_pred_proba):
        """
        Create visualizations for model evaluation
        
        Parameters:
        - X_test, y_test: Test data
        - y_pred_proba: Prediction probabilities
        """
        print("Creating visualizations...")
        
        # 1. Feature Importance Plot
        plt.figure(figsize=(12, 8))
        top_features = self.feature_importance.head(15)
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
        
        # 4. Threshold Optimization Plot
        thresholds_to_plot = np.arange(0.1, 0.9, 0.05)
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        for threshold in thresholds_to_plot:
            y_pred = (y_pred_proba > threshold).astype(int)
            f1_scores.append(f1_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred))
            recall_scores.append(recall_score(y_test, y_pred))
        
        plt.figure(figsize=(12, 8))
        plt.plot(thresholds_to_plot, f1_scores, 'b-', label='F1 Score')
        plt.plot(thresholds_to_plot, precision_scores, 'g-', label='Precision')
        plt.plot(thresholds_to_plot, recall_scores, 'r-', label='Recall')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Performance Metrics vs. Threshold')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/threshold_optimization.png')
        
        # 5. Confusion Matrix
        from sklearn.metrics import confusion_matrix
        y_pred = (y_pred_proba > self.best_threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('plots/confusion_matrix.png')
        
        print("Visualizations saved to 'plots' directory")
    
    def generate_shap_plots(self, X_test):
        """
        Generate SHAP value plots for model interpretation
        
        Parameters:
        - X_test: Test data features
        """
        try:
            import shap
            print("Generating SHAP plots for model interpretation...")
            
            # Create explainer
            explainer = shap.TreeExplainer(self.model)
            
            # Sample data for SHAP analysis (use subset if dataset is large)
            if X_test.shape[0] > 500:
                shap_sample = X_test[:500]
            else:
                shap_sample = X_test
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(shap_sample)
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, shap_sample, feature_names=self.feature_names, show=False)
            plt.title('SHAP Feature Importance')
            plt.tight_layout()
            plt.savefig('plots/shap_summary.png')
            plt.close()
            
            # Dependence plots for top features
            top_features_idx = np.argsort(np.abs(shap_values).mean(0))[-5:]
            for i in top_features_idx:
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(i, shap_values, shap_sample, feature_names=self.feature_names, show=False)
                plt.title(f'SHAP Dependence Plot: {self.feature_names[i]}')
                plt.tight_layout()
                plt.savefig(f'plots/shap_dependence_{self.feature_names[i]}.png')
                plt.close()
            
            print("SHAP plots saved to 'plots' directory")
            
        except ImportError:
            print("SHAP package not installed. Install with: pip install shap")
    
    def predict(self, data_path=None, threshold=None):
        """
        Make predictions on new data
        
        Parameters:
        - data_path: Path to new data
        - threshold: Probability threshold (uses optimized threshold if None)
        
        Returns:
        - Predictions dataframe
        """
        if self.model is None:
            print("Error: Model not trained. Please train the model first.")
            return None
        
        if threshold is None:
            threshold = self.best_threshold
        
        if data_path is None:
            # Use validation data
            print("Using validation data for prediction demonstration")
            X_train, X_test, y_train, y_test, _ = self.prepare_data()
            
            # Make predictions
            y_pred_proba = self.model.predict(X_test)
            y_pred = (y_pred_proba > threshold).astype(int)
            
            # Get account IDs (for demonstration)
            account_ids = np.arange(len(y_test))
            
        else:
            # Process new data
            print(f"Processing new data from {data_path}")
            new_reader = train_file_reader(data_path)
            X_new, _ = new_reader.get_transaction_merge_into_account()
            
            # Ensure same features as training
            missing_features = set(self.feature_names) - set(X_new.columns)
            for feature in missing_features:
                X_new[feature] = 0
            
            X_new = X_new[self.feature_names]
            
            # Scale features
            X_new_scaled = self.scaler.transform(X_new.values)
            
            # Make predictions
            y_pred_proba = self.model.predict(X_new_scaled)
            y_pred = (y_pred_proba > threshold).astype(int)
            
            # Get account IDs
            account_ids = new_reader.acc_info['ACCT_NBR'].values
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'account_id': account_ids,
            'fraud_probability': y_pred_proba,
            'fraud_prediction': y_pred
        })
        
        # Add risk categories
        results_df['risk_level'] = pd.cut(
            results_df['fraud_probability'], 
            bins=[0, 0.3, 0.6, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        
        # Print summary
        print(f"\nPrediction Summary:")
        print(f"Total accounts: {len(results_df)}")
        print(f"Flagged accounts: {results_df['fraud_prediction'].sum()} ({100*results_df['fraud_prediction'].mean():.2f}%)")
        print(f"Risk level distribution:")
        print(results_df['risk_level'].value_counts())
        
        # Save results
        results_df.to_csv('fraud_detection_results.csv', index=False)
        print("Results saved to 'fraud_detection_results.csv'")
        
        return results_df
    
    def run_pipeline(self):
        """
        Run the complete fraud detection pipeline
        """
        print("Running complete fraud detection pipeline...")
        
        # Step 1: Prepare data
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data()
        
        # Step 2: Train model
        model = self.train_model(X_train, y_train, X_test, y_test)
        
        # Step 3: Evaluate model
        metrics, y_pred_proba = self.evaluate_model(model, X_test, y_test)
        
        # Step 4: Cross-validate
        avg_precision, avg_recall, avg_f1 = self.cross_validate(
            np.vstack((X_train, X_test)), 
            np.hstack((y_train, y_test))
        )
        
        # Step 5: Visualize results
        self.visualize_results(X_test, y_test, y_pred_proba)
        
        # Step 6: Generate SHAP plots
        self.generate_shap_plots(X_test)
        
        # Step 7: Sample prediction
        self.predict()
        
        print("Pipeline completed successfully!")
        return metrics


def main():
    """
    Main function to run the fraud detection pipeline
    """
    # Specify your data path
    DATA_PATH = './comp_data/Train/'
    
    # Check for GPU availability
    use_gpu = True
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            print("GPU is available and will be used for training")
        else:
            print("GPU not available, using CPU instead")
    except ImportError:
        print("PyTorch not installed, cannot check GPU availability")
        use_gpu = False
    
    # Initialize and run the pipeline
    detector = LightGBMFraudDetector(data_path=DATA_PATH, use_gpu=use_gpu)
    metrics = detector.run_pipeline()
    
    print("\nFinal Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    main()