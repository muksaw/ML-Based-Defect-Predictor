#!/usr/bin/env python3
import unittest
import os
import json
import tempfile
import shutil
import pandas as pd
from unittest.mock import patch, MagicMock
import ml_harness

class TestMLHarness(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock config file
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        self.config = {
            "url_to_repo": "https://github.com/test/repo.git",
            "clone_repo_to": os.path.join(self.temp_dir, "repo"),
            "branch": "main",
            "from_date": "2023-01-01",
            "to_date": "2023-12-31",
            "confidence_threshold": 0.7
        }
        
        with open(self.config_path, "w") as f:
            json.dump(self.config, f)
        
        # Create a mock ground truth file
        self.ground_truth_path = os.path.join(self.temp_dir, "test_ground_truth.csv")
        self.ground_truth_data = pd.DataFrame({
            'modified_files': ['file1.py', 'file2.py', 'file3.py'],
            'is_buggy': [True, True, False]
        })
        self.ground_truth_data.to_csv(self.ground_truth_path, index=False)

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)

    def test_load_config(self):
        """Test loading configuration from file."""
        loaded_config = ml_harness.load_config(self.config_path)
        self.assertEqual(loaded_config, self.config)

    @patch('ml_harness.compare_with_ground_truth')
    def test_compare_with_ground_truth(self, mock_compare):
        """Test comparison with ground truth."""
        predictions = [
            {'file_path': 'file1.py', 'confidence': 0.9, 'is_buggy': True},
            {'file_path': 'file2.py', 'confidence': 0.6, 'is_buggy': False},
            {'file_path': 'file4.py', 'confidence': 0.8, 'is_buggy': True}
        ]
        
        # Set up the mock return value
        expected_metrics = {
            'precision': 0.5,
            'recall': 0.5,
            'f1_score': 0.5
        }
        mock_compare.return_value = expected_metrics
        
        # Call the function with our test data
        metrics = ml_harness.compare_with_ground_truth(predictions, self.ground_truth_path)
        
        # Verify the mock was called with the correct arguments
        mock_compare.assert_called_once_with(predictions, self.ground_truth_path)
        
        # Verify the return value
        self.assertEqual(metrics, expected_metrics)

    def test_print_predictions(self):
        """Test printing predictions (basic functionality test)."""
        predictions = [
            {'file_path': 'file1.py', 'confidence': 0.9, 'is_buggy': True},
            {'file_path': 'file2.py', 'confidence': 0.8, 'is_buggy': True},
            {'file_path': 'file3.py', 'confidence': 0.3, 'is_buggy': False}
        ]
        
        # This just checks that the function doesn't raise any errors
        try:
            ml_harness.print_predictions(predictions)
            test_passed = True
        except Exception as e:
            test_passed = False
        
        self.assertTrue(test_passed)

    @patch('argparse.ArgumentParser.parse_args')
    @patch('ml_harness.load_config')
    @patch('ml_defect_predictor.MLDefectPredictor')
    def test_main_train_and_predict(self, mock_predictor_class, mock_load_config, mock_parse_args):
        """Test main function with train and predict flags."""
        # Set up mocks
        mock_args = MagicMock()
        mock_args.config = self.config_path
        mock_args.train = True
        mock_args.predict = True
        mock_args.save_model = False
        mock_args.load_model = False
        mock_args.model_path = "test_model.joblib"
        mock_args.ground_truth = self.ground_truth_path
        mock_parse_args.return_value = mock_args
        
        mock_load_config.return_value = self.config
        
        mock_predictor = MagicMock()
        mock_predictor_class.return_value = mock_predictor
        
        # Set up training metrics
        mock_predictor.train.return_value = {
            'precision': 0.8,
            'recall': 0.7,
            'f1_score': 0.75,
            'buggy_files_count': 2,
            'total_files_count': 10,
            'feature_importances': {'num_commits': 0.5, 'bug_fix_count': 0.3}
        }
        
        # Set up prediction results
        mock_predictor.predict.return_value = [
            {'file_path': 'file1.py', 'confidence': 0.9, 'is_buggy': True},
            {'file_path': 'file2.py', 'confidence': 0.8, 'is_buggy': True}
        ]
        
        # Call the main function
        ml_harness.main()
        
        # Verify the correct calls were made
        mock_predictor_class.assert_called_once_with(self.config)
        mock_predictor.train.assert_called_once()
        mock_predictor.predict.assert_called_once()
        
        # Verify save_model wasn't called since save_model flag is False
        mock_predictor.save_model.assert_not_called()

    @patch('argparse.ArgumentParser.parse_args')
    @patch('ml_harness.load_config')
    @patch('ml_defect_predictor.MLDefectPredictor')
    def test_main_load_and_predict(self, mock_predictor_class, mock_load_config, mock_parse_args):
        """Test main function with load and predict flags."""
        # Set up mocks
        mock_args = MagicMock()
        mock_args.config = self.config_path
        mock_args.train = False
        mock_args.predict = True
        mock_args.save_model = False
        mock_args.load_model = True
        mock_args.model_path = "test_model.joblib"
        mock_args.ground_truth = self.ground_truth_path
        mock_parse_args.return_value = mock_args
        
        mock_load_config.return_value = self.config
        
        mock_predictor = MagicMock()
        mock_predictor_class.return_value = mock_predictor
        mock_predictor.load_model.return_value = True
        
        # Set up prediction results
        mock_predictor.predict.return_value = [
            {'file_path': 'file1.py', 'confidence': 0.9, 'is_buggy': True},
            {'file_path': 'file2.py', 'confidence': 0.8, 'is_buggy': True}
        ]
        
        # Call the main function
        ml_harness.main()
        
        # Verify the correct calls were made
        mock_predictor_class.assert_called_once_with(self.config)
        mock_predictor.load_model.assert_called_once_with("test_model.joblib")
        mock_predictor.train.assert_not_called()  # Shouldn't train
        mock_predictor.predict.assert_called_once()

if __name__ == "__main__":
    unittest.main() 