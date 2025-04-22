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
        
        # Time weighting parameters
        self.time_decay_factor = config.get("time_decay_factor", 30.0)  # Default: 30 days half-life
        
        # Separate file extensions into code and documentation
        self.code_extensions = ['.py', '.java', '.js', '.cpp', '.c', '.h', '.cs', '.ts']
        self.doc_extensions = ['.md', '.rst', '.txt']
        self.file_extensions = config.get("file_extensions", self.code_extensions)
        
        # Ensure the clone directory exists
        os.makedirs(self.clone_repo_to, exist_ok=True)
        
        # Test files exclusion patterns
#       self.test_keywords = ['test', 'tests', '_test', 'test_']
        
        # ML model
        self.model = None
        self.scaler = StandardScaler()
        
        # Cache for extracted features
        self._feature_cache = None
        self._bug_fixes_cache = None

    def is_code_file(self, file_path):
        """
        Check if a file is a code file that should be analyzed for bugs.
        
        Args:
            file_path (str): Path to the file
        
        Returns:
            bool: True if the file should be analyzed for bugs
        """
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.code_extensions
        
    def is_test_file(self, file_path):
        """
        Check if a file is a test file that should be excluded from analysis.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            bool: True if the file is a test file
        """
        # Check if file path contains test indicators
        test_indicators = ['test', 'tests', '_test', 'test_', '/test/', '/tests/']
        file_path_lower = file_path.lower()
        
        # Check if it's in a test directory or has test in the name
        for indicator in test_indicators:
            if indicator in file_path_lower:
                return True
                
        return False
        
    def calculate_time_weight(self, commit_date, current_date):
        """
        Calculate a time weight for a commit based on its age.
        More recent commits get higher weight.
        
        Args:
            commit_date: Date of the commit
            current_date: Current date (reference point)
            
        Returns:
            float: Weight between 0 and 1 (higher for more recent commits)
        """
        # Make dates timezone-naive for comparison
        commit_date = commit_date.replace(tzinfo=None) if commit_date.tzinfo else commit_date
        current_date = current_date.replace(tzinfo=None) if hasattr(current_date, 'tzinfo') and current_date.tzinfo else current_date
        
        # Calculate days since commit
        days_since_commit = (current_date - commit_date).days
        
        # Apply exponential decay with half-life of time_decay_factor days
        weight = math.exp(-days_since_commit / self.time_decay_factor)
        
        return weight

    def calculate_relative_risk(self, file_metrics):
        """
        Calculate relative risk scores for files based on repository averages.
        
        Args:
            file_metrics (dict): Dictionary of file metrics
            
        Returns:
            dict: Updated file metrics with relative risk scores
        """
        if not file_metrics:
            return file_metrics
            
        # Get repository averages
        commit_counts = [m['n_commits'] for m in file_metrics.values()]
        bug_fix_counts = [m['bug_fix_count'] for m in file_metrics.values()]
        complexity_values = [m['total_complexity'] / max(1, m['n_functions']) for m in file_metrics.values()]
        
        avg_commits = np.mean(commit_counts) if commit_counts else 0
        avg_bug_fixes = np.mean(bug_fix_counts) if bug_fix_counts else 0
        avg_complexity = np.mean(complexity_values) if complexity_values else 0
        
        # Avoid division by zero
        if avg_commits == 0:
            avg_commits = 1
        if avg_complexity == 0:
            avg_complexity = 1
            
        # Calculate relative scores
        for file_path, metrics in file_metrics.items():
            # Bug fix density (normalized by commits)
            if metrics['n_commits'] > 0:
                bug_ratio = metrics['bug_fix_count'] / metrics['n_commits']
            else:
                bug_ratio = 0
                
            # Calculate relative values (how much above/below average)
            commit_factor = metrics['n_commits'] / avg_commits
            
            # Complexity factor
            file_complexity = metrics['total_complexity'] / max(1, metrics['n_functions'])
            complexity_factor = file_complexity / avg_complexity
            
            # Calculate weighted bug density
            weighted_bug_density = bug_ratio * (1 + 0.5 * (complexity_factor - 1))
            
            # Calculate relative risk score
            relative_risk = weighted_bug_density * (1 + 0.3 * (commit_factor - 1))
            
            # Add to metrics
            metrics['bug_ratio'] = bug_ratio
            metrics['relative_risk'] = relative_risk
            
            # Add risk category
            if relative_risk < 0.5:
                risk_category = "Low"
            elif relative_risk < 1.0:
                risk_category = "Medium-Low"
            elif relative_risk < 1.5:
                risk_category = "Medium"
            elif relative_risk < 3.0:
                risk_category = "Medium-High"
            else:
                risk_category = "High"
                
            metrics['risk_category'] = risk_category
            
        return file_metrics

    def adapt_confidence_threshold(self):
        """
        Adapt confidence threshold based on repository characteristics.
        
        Returns:
            float: Adapted confidence threshold
        """
        if not self.from_date or not self.to_date:
            return self.confidence_threshold
            
        # Get time span in days
        time_span = (self.to_date - self.from_date).days
        
        # Base adjustment on time span (longer range = higher threshold)
        # For spans over 1 year, increase threshold to reduce false positives
        base_adjustment = min(0.2, (time_span / 365) * 0.1)  # Max +0.2 for very long ranges
        
        adjusted_threshold = self.confidence_threshold + base_adjustment
        
        logger.info(f"Adjusted confidence threshold from {self.confidence_threshold} to {adjusted_threshold} based on time span of {time_span} days")
        
        return adjusted_threshold

    def extract_features(self, historical_data=True, max_commits=6000):
        """
        Extract features from the Git repository for training or prediction.
        
        Args:
            historical_data (bool): If True, returns data for training including labels
                                  If False, returns only features for prediction
            max_commits (int): Maximum number of commits to analyze
        
        Returns:
            DataFrame: Features data for training or prediction
        """
        # If we have cached features and we're not requesting historical data,
        # return the cached features without labels
        if self._feature_cache is not None and not historical_data:
            logger.info("Using cached features for prediction")
            return self._feature_cache
            
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
            
            # Reference date for time weighting (use to_date or current date)
            reference_date = self.to_date if self.to_date else datetime.now()
            
            # Process all commits to extract metrics
            for i, commit in enumerate(commits):
                if i % 100 == 0:
                    logger.info(f"Processing commit {i+1}/{total_commits}...")
                
                # Calculate time weight for this commit (higher for more recent commits)
                time_weight = self.calculate_time_weight(commit.author_date, reference_date)
                
                # Check if this is a bug fix commit
                is_bug_fix = self.is_bug_fix(commit.msg)
                
                # Process each modified file
                for file in commit.modified_files:
                    file_path = file.new_path or file.old_path
                    if file_path is None:
                        continue
                        
                    # Skip if not a code file
                    if not self.is_code_file(file_path):
                        continue
                    
                    # Skip test files
                    if self.is_test_file(file_path):
                        continue
                    
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
                            'weighted_commits': 0,
                            'weighted_bugs': 0,
                            'commit_dates': [],
                            'is_buggy': False
                        }
                    
                    # Update metrics with time weighting
                    file_metrics[file_path]['n_commits'] += 1
                    file_metrics[file_path]['weighted_commits'] += time_weight
                    file_metrics[file_path]['n_authors'].add(commit.author.email)
                    file_metrics[file_path]['n_lines_added'] += file.added_lines
                    file_metrics[file_path]['n_lines_deleted'] += file.deleted_lines
                    file_metrics[file_path]['commit_dates'].append(commit.author_date)
                    
                    # Update complexity metrics if available
                    if file.complexity is not None:
                        file_metrics[file_path]['total_complexity'] += file.complexity
                        file_metrics[file_path]['n_functions'] += 1
                    
                    # Update modification timestamps
                    if commit.author_date < file_metrics[file_path]['first_modified']:
                        file_metrics[file_path]['first_modified'] = commit.author_date
                    if commit.author_date > file_metrics[file_path]['last_modified']:
                        file_metrics[file_path]['last_modified'] = commit.author_date
                    
                    # Mark files in bug fix commits with time weighting
                    if is_bug_fix:
                        file_metrics[file_path]['bug_fix_count'] += 1
                        file_metrics[file_path]['weighted_bugs'] += time_weight
                        if historical_data:
                            bug_fixes.add(file_path)
            
            logger.info(f"Finished analyzing {total_commits} commits")
            
            # Calculate relative risk scores
            file_metrics = self.calculate_relative_risk(file_metrics)
            
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
                
                # Calculate commit density (commits per month)
                if age_days > 0:
                    commit_density = (metrics['n_commits'] / age_days) * 30
                else:
                    commit_density = 0
                
                # Create a feature vector
                features = {
                    'file_path': file_path,
                    'n_commits': metrics['n_commits'],
                    'weighted_commits': metrics['weighted_commits'],
                    'n_authors': len(metrics['n_authors']),
                    'n_lines_added': metrics['n_lines_added'],
                    'n_lines_deleted': metrics['n_lines_deleted'],
                    'avg_complexity': avg_complexity,
                    'age_days': age_days,
                    'recent_modified_days': recent_modified_days,
                    'bug_fix_count': metrics['bug_fix_count'],
                    'weighted_bugs': metrics['weighted_bugs'],
                    'commit_density': commit_density,
                    'relative_risk': metrics.get('relative_risk', 0),
                    'risk_category': metrics.get('risk_category', 'Unknown')
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
                cols = ['file_path', 'n_commits', 'weighted_commits', 'n_authors', 'n_lines_added', 
                       'n_lines_deleted', 'avg_complexity', 'age_days', 'recent_modified_days', 
                       'bug_fix_count', 'weighted_bugs', 'commit_density', 'relative_risk', 'risk_category']
                if historical_data:
                    cols.append('is_buggy')
                return pd.DataFrame(columns=cols)
            
            # Export features to CSV with more descriptive name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
            os.makedirs(output_dir, exist_ok=True)
            
            # Add analysis type to filename
            analysis_type = 'historical' if historical_data else 'prediction'
            csv_path = os.path.join(output_dir, f'feature_table_{analysis_type}_{timestamp}.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"Feature table exported to {csv_path}")
            
            # Cache the features without labels for future use
            if historical_data:
                self._feature_cache = df.drop('is_buggy', axis=1) if 'is_buggy' in df.columns else df
                self._bug_fixes_cache = bug_fixes
                
            logger.info(f"Extracted features for {len(df)} files")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Create an empty DataFrame with the expected columns
            cols = ['file_path', 'n_commits', 'weighted_commits', 'n_authors', 'n_lines_added', 
                   'n_lines_deleted', 'avg_complexity', 'age_days', 'recent_modified_days', 
                   'bug_fix_count', 'weighted_bugs', 'commit_density', 'relative_risk', 'risk_category']
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
        
        # Select features for the model (only numerical features)
        feature_cols = ['n_commits', 'weighted_commits', 'n_authors', 'n_lines_added', 
                        'n_lines_deleted', 'avg_complexity', 'age_days', 'recent_modified_days', 
                        'bug_fix_count', 'weighted_bugs', 'commit_density', 'relative_risk']
        
        # Don't include categorical features like 'risk_category' as the model needs numeric values
        # Filter to include only columns that exist in the DataFrame
        feature_cols = [col for col in feature_cols if col in df.columns]
        
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
        
        # Get feature importances (only for numerical features that were used in training)
        feature_cols = ['n_commits', 'weighted_commits', 'n_authors', 'n_lines_added', 
                        'n_lines_deleted', 'avg_complexity', 'age_days', 'recent_modified_days', 
                        'bug_fix_count', 'weighted_bugs', 'commit_density', 'relative_risk']
        
        # Ensure we only include features that were actually used
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Map feature importances to feature names
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
        if len(self.model.classes_) < 2:
            logger.error("Model was trained with only one class. Cannot compute probability for both classes.")
            return []
        y_prob = self.model.predict_proba(X_scaled)[:, 1]  # Probability of being buggy
        
        # Get adaptive threshold based on time range
        adaptive_threshold = self.adapt_confidence_threshold()
        
        # Create results
        results = []
        for i, file_path in enumerate(file_paths):
            confidence = float(y_prob[i])
            
            # Include the relative risk and category from original features
            relative_risk = float(df[df['file_path'] == file_path]['relative_risk'].values[0]) if 'relative_risk' in df.columns else 0
            risk_category = str(df[df['file_path'] == file_path]['risk_category'].values[0]) if 'risk_category' in df.columns else 'Unknown'
            
            # Only include predictions above the adaptive confidence threshold
            if confidence >= adaptive_threshold:
                results.append({
                    "file_path": file_path,
                    "is_buggy": bool(y_pred[i]),
                    "confidence": confidence,
                    "relative_risk": relative_risk,
                    "risk_category": risk_category,
                    "_original_threshold": self.confidence_threshold,
                    "_adjusted_threshold": adaptive_threshold,
                    "_time_span": (self.to_date - self.from_date).days if self.to_date and self.from_date else 0
                })
        
        # Sort by confidence (highest first)
        results = sorted(results, key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"Generated predictions for {len(results)} files above confidence threshold {adaptive_threshold}")
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

    def is_bug_fix(self, commit_msg):
        # Add more sophisticated logic analysis
        return any(keyword in commit_msg.lower() for keyword in [
            'fix', 'bug', 'issue', 'error', 'crash', 'problem',
            'defect', 'fault', 'flaw', 'incorrect', 'regression'
        ]) 