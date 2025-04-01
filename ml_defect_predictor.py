import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pydriller import Repository, Git
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import joblib
import logging
import concurrent.futures
import math
import time
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLDefectPredictor:
    """
    Machine Learning-based Defect Predictor that analyzes Git repository history
    to identify potentially buggy files based on various metrics.
    """
    
    def __init__(self, config):
        """
        Initialize the ML Defect Predictor.
        
        Args:
            config (dict): Configuration dictionary containing repository info
        """
        self.url_to_repo = config["url_to_repo"]
        self.clone_repo_to = config["clone_repo_to"]
        self.branch = config.get("branch", "master")
        self.from_date = datetime.strptime(config["from_date"], "%Y-%m-%d") if "from_date" in config else None
        self.to_date = datetime.strptime(config["to_date"], "%Y-%m-%d") if "to_date" in config else None
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.file_extensions = config.get("file_extensions", ['.py', '.java', '.js', '.cpp', '.c', '.h', '.cs'])
        
        # Ensure the clone directory exists
        os.makedirs(self.clone_repo_to, exist_ok=True)

        # Test files exclusion patterns
#       self.test_keywords = ['test', 'tests', '_test', 'test_']
        
        # ML model
        self.model = None
        self.scaler = StandardScaler()

#    def is_test_file(self, file_path):
#        """
#        Check if a file is likely a test file based on its name or path.
#        
#        Args:
#            file_path (str): Path to the file
#        
#        Returns:
#            bool: True if the file is identified as a test file, otherwise False
#        """
#        return any(keyword in file_path.lower() for keyword in self.test_keywords)
        
    def extract_features(self, historical_data=True, max_commits=5000):
        """
        Extract features from the Git repository for model training or prediction.
        
        Args:
            historical_data (bool): If True, returns data for training including labels
                                  If False, returns only features for prediction
            max_commits (int): Maximum number of commits to analyze
        
        Returns:
            DataFrame: Features data for training or prediction
        """
        logger.info(f"Extracting features from repository: {self.url_to_repo}")
        
        # Initialize data structures
        file_metrics = {}
        bug_fixes = set()
        
        try:
            # Clone the repository if needed and collect commits
            logger.info(f"Cloning repository to {self.clone_repo_to} if not already cloned...")
            repo = Repository(self.url_to_repo, 
                            clone_repo_to=self.clone_repo_to,
                            only_in_branch=self.branch,
                            since=self.from_date,
                            to=self.to_date)
            
            # Get commits to analyze
            logger.info("Collecting commits for analysis...")
            
            # Limit the number of commits to process to avoid extremely long run times
            commits = list(repo.traverse_commits())[:max_commits]
            total_commits = len(commits)
            logger.info(f"Found {total_commits} commits to analyze (limited to {max_commits})")
            
            # Process all commits to extract metrics
            for i, commit in enumerate(commits):
                if i % 100 == 0:
                    logger.info(f"Processing commit {i+1}/{total_commits}...")
                
                # Check if this is a bug fix commit (simple heuristic)
                is_bug_fix = any(keyword in commit.msg.lower() for keyword in 
                                ['fix', 'bug', 'issue', 'error', 'crash', 'problem',
                                'defect', 'fault', 'flaw', 'incorrect', 'regression'])
                
                # Process each modified file
                for file in commit.modified_files:
                    file_ext = os.path.splitext(file.filename)[1]
                    if file_ext not in self.file_extensions:
                        continue  # Skip non-code files
                    
                    file_path = file.new_path or file.old_path
                    if file_path is None:
                        continue

                     # Skip test files
#                    if self.is_test_file(file_path):
#                        continue
                    
                    # Initialize file data if it doesn't exist
                    if file_path not in file_metrics:
                        file_metrics[file_path] = {
                            'n_commits': 0,
                            'n_authors': set(),
                            'n_lines_added': 0,
                            'n_lines_deleted': 0,
                            'avg_complexity': 0,
                            'total_complexity': 0,
                            'n_functions': 0,
                            'last_modified': commit.author_date,
                            'first_modified': commit.author_date,
                            'bug_fix_count': 0,
                            'is_buggy': False  # Will be updated if historical_data=True
                        }
                    
                    # Update metrics
                    file_metrics[file_path]['n_commits'] += 1
                    file_metrics[file_path]['n_authors'].add(commit.author.email)
                    file_metrics[file_path]['n_lines_added'] += file.added_lines
                    file_metrics[file_path]['n_lines_deleted'] += file.deleted_lines
                    
                    # Update complexity metrics if available
                    if file.complexity is not None:
                        file_metrics[file_path]['total_complexity'] += file.complexity
                        file_metrics[file_path]['n_functions'] += 1
                    
                    # Update modification timestamps
                    if commit.author_date < file_metrics[file_path]['first_modified']:
                        file_metrics[file_path]['first_modified'] = commit.author_date
                    if commit.author_date > file_metrics[file_path]['last_modified']:
                        file_metrics[file_path]['last_modified'] = commit.author_date
                    
                    # Mark files in bug fix commits
                    if is_bug_fix:
                        file_metrics[file_path]['bug_fix_count'] += 1
                        if historical_data:
                            bug_fixes.add(file_path)
            
            logger.info(f"Finished analyzing {total_commits} commits")
            
            # Create DataFrame from metrics
            data = []
            for file_path, metrics in file_metrics.items():
                # Calculate derived metrics
                # Make both dates timezone-naive for consistent comparison
                first_modified = metrics['first_modified'].replace(tzinfo=None) if metrics['first_modified'].tzinfo else metrics['first_modified']
                last_modified = metrics['last_modified'].replace(tzinfo=None) if metrics['last_modified'].tzinfo else metrics['last_modified']
                to_date_naive = self.to_date.replace(tzinfo=None) if self.to_date and hasattr(self.to_date, 'tzinfo') and self.to_date.tzinfo else self.to_date
                
                age_days = (to_date_naive - first_modified).days if to_date_naive else 0
                recent_modified_days = (to_date_naive - last_modified).days if to_date_naive else 0
                
                # Calculate average complexity
                avg_complexity = 0
                if metrics['n_functions'] > 0:
                    avg_complexity = metrics['total_complexity'] / metrics['n_functions']
                
                # Create a feature vector
                features = {
                    'file_path': file_path,
                    'n_commits': metrics['n_commits'],
                    'n_authors': len(metrics['n_authors']),
                    'n_lines_added': metrics['n_lines_added'],
                    'n_lines_deleted': metrics['n_lines_deleted'],
                    'avg_complexity': avg_complexity,
                    'age_days': age_days,
                    'recent_modified_days': recent_modified_days,
                    'bug_fix_count': metrics['bug_fix_count'],
                }
                
                # Add label for training data
                if historical_data:
                    features['is_buggy'] = 1 if file_path in bug_fixes else 0
                
                data.append(features)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            if len(df) == 0:
                logger.warning("No files extracted from repository")
                # Create an empty DataFrame with the expected columns
                cols = ['file_path', 'n_commits', 'n_authors', 'n_lines_added', 
                       'n_lines_deleted', 'avg_complexity', 'age_days', 
                       'recent_modified_days', 'bug_fix_count']
                if historical_data:
                    cols.append('is_buggy')
                return pd.DataFrame(columns=cols)
                
            logger.info(f"Extracted features for {len(df)} files")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Create an empty DataFrame with the expected columns
            cols = ['file_path', 'n_commits', 'n_authors', 'n_lines_added', 
                   'n_lines_deleted', 'avg_complexity', 'age_days', 
                   'recent_modified_days', 'bug_fix_count']
            if historical_data:
                cols.append('is_buggy')
            return pd.DataFrame(columns=cols)
    
    def train_model(self, X_train, y_train):
        """
        Train a Random Forest classifier on the given data.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info("Training defect prediction model")
        
        # Use Random Forest as our classifier
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        logger.info("Model training completed")
    
    def prepare_data(self, df, for_training=True):
        """
        Prepare the data for training or prediction.
        
        Args:
            df: DataFrame with features
            for_training: If True, splits data for training, otherwise prepares for prediction
            
        Returns:
            Training split or prediction ready data
        """
        if len(df) == 0:
            logger.warning("Empty DataFrame provided for data preparation")
            return None
            
        # Extract file paths
        file_paths = df['file_path'].values
        
        # Select features for the model
        feature_cols = ['n_commits', 'n_authors', 'n_lines_added', 'n_lines_deleted',
                       'avg_complexity', 'age_days', 'recent_modified_days', 'bug_fix_count']
        
        # Replace NaN values with 0
        X = df[feature_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X) if for_training else self.scaler.transform(X)
        
        if for_training:
            # Get labels
            y = df['is_buggy'].values
            
            # Split data
            X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
                X_scaled, y, file_paths, test_size=0.3, random_state=42
            )
            
            return X_train, X_test, y_train, y_test, paths_train, paths_test
        else:
            return X_scaled, file_paths
    
    def train(self):
        """
        Extract features and train the model on historical data.
        
        Returns:
            dict: Training metrics
        """
        # Extract features with historical data
        df = self.extract_features(historical_data=True)
        
        if len(df) == 0:
            logger.error("No data available for training")
            return {"error": "No data available for training"}
            
        # Prepare data for training
        data = self.prepare_data(df, for_training=True)
        if data is None:
            return {"error": "Failed to prepare training data"}
            
        X_train, X_test, y_train, y_test, paths_train, paths_test = data
        
        # Train the model
        self.train_model(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )
        
        # Get feature importances
        feature_cols = ['n_commits', 'n_authors', 'n_lines_added', 'n_lines_deleted',
                       'avg_complexity', 'age_days', 'recent_modified_days', 'bug_fix_count']
        importances = dict(zip(feature_cols, self.model.feature_importances_))
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "feature_importances": importances,
            "buggy_files_count": int(sum(y_test)),
            "total_files_count": len(y_test)
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    
    def predict(self):
        """
        Predict buggy files in the current state of the repository.
        
        Returns:
            list: Predictions with file paths and confidence scores
        """
        # Extract features for prediction (no historical labeling)
        df = self.extract_features(historical_data=False)
        
        if len(df) == 0:
            logger.error("No files to predict on")
            return []
            
        # Prepare data for prediction
        data = self.prepare_data(df, for_training=False)
        if data is None:
            return []
            
        X_scaled, file_paths = data
        
        # Check if model exists
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return []
        
        # Get predictions and probabilities
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]  # Probability of being buggy
        
        # Create results
        results = []
        for i, file_path in enumerate(file_paths):
            confidence = float(y_prob[i])
            # Only include predictions above the confidence threshold
            if confidence >= self.confidence_threshold:
                results.append({
                    "file_path": file_path,
                    "is_buggy": bool(y_pred[i]),
                    "confidence": confidence
                })
        
        # Sort by confidence (highest first)
        results = sorted(results, key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"Generated predictions for {len(results)} files above confidence threshold")
        return results
    
    def save_model(self, path="ml_defect_model.joblib"):
        """
        Save the trained model to a file.
        
        Args:
            path (str): Path to save the model
        """
        if self.model is None:
            logger.error("No model to save. Train the model first.")
            return False
            
        model_data = {
            "model": self.model,
            "scaler": self.scaler
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
        return True
    
    def load_model(self, path="ml_defect_model.joblib"):
        """
        Load a trained model from a file.
        
        Args:
            path (str): Path to the saved model
        """
        try:
            model_data = joblib.load(path)
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False 