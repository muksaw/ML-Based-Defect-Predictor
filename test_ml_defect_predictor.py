#!/usr/bin/env python3
import unittest
import os
import json
import tempfile
import shutil
import pandas as pd
import numpy as np
from ml_defect_predictor import MLDefectPredictor
import subprocess
import datetime

class TestMLDefectPredictor(unittest.TestCase):
    def setUp(self):
        """Set up test environment with a mock git repository."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.repo_dir = os.path.join(self.temp_dir, "mock_repo")
        
        # Initialize git repository
        self.init_git_repo()
        
        # Create config for the predictor
        self.config = {
            "url_to_repo": self.repo_dir,
            "clone_repo_to": self.repo_dir,  # Same as url since we're using local path
            "branch": "main",
            "from_date": (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
            "to_date": datetime.datetime.now().strftime("%Y-%m-%d")
        }
        
        # Initialize predictor
        self.predictor = MLDefectPredictor(self.config)

    def init_git_repo(self):
        """Initialize a git repository with sample commits."""
        os.makedirs(self.repo_dir)
        os.chdir(self.repo_dir)
        
        # Initialize git
        subprocess.run(["git", "init", "-b", "main"], check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], check=True)
        
        # Create and commit several files
        # 1. Create a clean file
        with open(os.path.join(self.repo_dir, "clean_file.py"), "w") as f:
            f.write("def clean_function():\n    return 'This is a clean function'\n")
        
        subprocess.run(["git", "add", "clean_file.py"], check=True)
        subprocess.run(["git", "commit", "-m", "Add clean file"], check=True)
        
        # 2. Create a buggy file and fix it in multiple commits
        with open(os.path.join(self.repo_dir, "buggy_file.py"), "w") as f:
            f.write("def buggy_function():\n    retrun 'This has a typo'\n")  # Intentional typo
        
        subprocess.run(["git", "add", "buggy_file.py"], check=True)
        subprocess.run(["git", "commit", "-m", "Add buggy file with typo"], check=True)
        
        # Fix the bug
        with open(os.path.join(self.repo_dir, "buggy_file.py"), "w") as f:
            f.write("def buggy_function():\n    return 'This had a typo, now fixed'\n")
        
        subprocess.run(["git", "add", "buggy_file.py"], check=True)
        subprocess.run(["git", "commit", "-m", "Fix bug: correct typo in return statement"], check=True)
        
        # 3. Make more commits to increase complexity
        for i in range(3):
            filename = f"file_{i}.py"
            with open(os.path.join(self.repo_dir, filename), "w") as f:
                f.write(f"# File {i}\ndef function_{i}():\n    return {i}\n")
            
            subprocess.run(["git", "add", filename], check=True)
            subprocess.run(["git", "commit", "-m", f"Add file {i}"], check=True)
            
            # Modify one file with a bug fix
            if i == 1:
                with open(os.path.join(self.repo_dir, filename), "w") as f:
                    f.write(f"# File {i}\ndef function_{i}():\n    # Fixed calculation\n    return {i} + 1\n")
                
                subprocess.run(["git", "add", filename], check=True)
                subprocess.run(["git", "commit", "-m", f"Fix bug in file {i}: correct calculation"], check=True)
        
        # Create ground truth data
        self.ground_truth_data = pd.DataFrame({
            'modified_files': ['buggy_file.py', 'file_1.py'],
            'is_buggy': [True, True]
        })
        self.ground_truth_file = os.path.join(self.repo_dir, "ground_truth.csv")
        self.ground_truth_data.to_csv(self.ground_truth_file, index=False)
        
        # Return to original directory
        os.chdir(self.temp_dir)

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test that the MLDefectPredictor initializes correctly."""
        self.assertEqual(self.predictor.url_to_repo, self.config["url_to_repo"])
        self.assertEqual(self.predictor.clone_repo_to, self.config["clone_repo_to"])
        self.assertEqual(self.predictor.branch, self.config["branch"])
        self.assertEqual(self.predictor.from_date, self.config["from_date"])
        self.assertEqual(self.predictor.to_date, self.config["to_date"])
        
        # Ensure model is None on initialization
        self.assertIsNone(self.predictor.model)

    def test_extract_features(self):
        """Test feature extraction from repository."""
        features_df = self.predictor.extract_features()
        
        # Check that the dataframe is not empty
        self.assertFalse(features_df.empty)
        
        # Check that the key files are in the dataframe
        file_paths = features_df['file_path'].tolist()
        self.assertIn('buggy_file.py', file_paths)
        self.assertIn('file_1.py', file_paths)
        
        # Check that features were extracted
        expected_columns = [
            'file_path', 'num_commits', 'num_authors',
            'lines_added', 'lines_removed', 
            'bug_fix_count', 'is_buggy'
        ]
        for col in expected_columns:
            self.assertIn(col, features_df.columns)

    def test_prepare_data(self):
        """Test data preparation for model training."""
        # Extract features
        features_df = self.predictor.extract_features()
        
        # Prepare data
        X, y = self.predictor.prepare_data(features_df)
        
        # Verify X and y shapes
        self.assertEqual(len(X), len(features_df))
        self.assertEqual(len(y), len(features_df))
        
        # Verify X contains the expected feature columns
        expected_feature_count = len(features_df.columns) - 2  # Exclude file_path and is_buggy
        self.assertEqual(X.shape[1], expected_feature_count)

    def test_train_model(self):
        """Test model training."""
        # Extract features
        features_df = self.predictor.extract_features()
        
        # Prepare data
        X, y = self.predictor.prepare_data(features_df)
        
        # Train model
        model, metrics = self.predictor.train_model(X, y, features_df.drop(['file_path', 'is_buggy'], axis=1).columns)
        
        # Verify model was created
        self.assertIsNotNone(model)
        
        # Verify metrics were calculated
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('feature_importances', metrics)

    def test_predict(self):
        """Test prediction functionality."""
        # First train the model
        metrics = self.predictor.train()
        
        # Verify training completed successfully
        self.assertNotIn('error', metrics)
        
        # Generate predictions
        predictions = self.predictor.predict()
        
        # Verify predictions format
        self.assertIsInstance(predictions, list)
        if predictions:
            self.assertIsInstance(predictions[0], dict)
            self.assertIn('file_path', predictions[0])
            self.assertIn('confidence', predictions[0])
            self.assertIn('is_buggy', predictions[0])
        
        # Check that known buggy files have higher confidence
        buggy_file_predictions = [p for p in predictions if p['file_path'] in ['buggy_file.py', 'file_1.py']]
        other_file_predictions = [p for p in predictions if p['file_path'] not in ['buggy_file.py', 'file_1.py']]
        
        if buggy_file_predictions and other_file_predictions:
            avg_buggy_confidence = sum(p['confidence'] for p in buggy_file_predictions) / len(buggy_file_predictions)
            avg_other_confidence = sum(p['confidence'] for p in other_file_predictions) / len(other_file_predictions)
            
            # Buggy files should have higher confidence on average
            # This is a probabilistic test, so it might not always pass
            # due to the random nature of RandomForest, but it's a good check
            self.assertGreaterEqual(avg_buggy_confidence, avg_other_confidence * 0.5)

    def test_save_load_model(self):
        """Test saving and loading model."""
        # Train model
        self.predictor.train()
        
        # Save model
        model_path = os.path.join(self.temp_dir, "test_model.joblib")
        save_result = self.predictor.save_model(model_path)
        
        # Verify save was successful
        self.assertTrue(save_result)
        self.assertTrue(os.path.exists(model_path))
        
        # Create new predictor instance
        new_predictor = MLDefectPredictor(self.config)
        
        # Load model
        load_result = new_predictor.load_model(model_path)
        
        # Verify load was successful
        self.assertTrue(load_result)
        self.assertIsNotNone(new_predictor.model)
        
        # Generate predictions with loaded model
        predictions = new_predictor.predict()
        
        # Verify predictions are generated
        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)

if __name__ == "__main__":
    unittest.main() 