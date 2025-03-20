#!/usr/bin/env python3
import os
import json
import pandas as pd
import argparse
from ml_defect_predictor import MLDefectPredictor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_file):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)

def compare_with_ground_truth(predicted_files, ground_truth_file):
    """
    Compare predicted buggy files with ground truth data.
    
    Args:
        predicted_files (list): List of dictionaries with predicted buggy files
        ground_truth_file (str): Path to the ground truth CSV file
    
    Returns:
        dict: Comparison metrics
    """
    try:
        # Load ground truth
        df = pd.read_csv(ground_truth_file)
        ground_truth_files = set(df['modified_files'].dropna().tolist())
        
        # Get predicted files (those with is_buggy=True)
        predicted_files_set = {f['file_path'] for f in predicted_files if f['is_buggy']}
        
        # Get top N files by confidence
        top_5_files = {f['file_path'] for f in predicted_files[:5]} if len(predicted_files) >= 5 else predicted_files_set
        top_10_files = {f['file_path'] for f in predicted_files[:10]} if len(predicted_files) >= 10 else predicted_files_set
        
        # Calculate metrics
        true_positives = predicted_files_set.intersection(ground_truth_files)
        false_positives = predicted_files_set - ground_truth_files
        false_negatives = ground_truth_files - predicted_files_set
        
        # Top N metrics
        top_5_true_positives = top_5_files.intersection(ground_truth_files)
        top_10_true_positives = top_10_files.intersection(ground_truth_files)
        
        # Calculate precision, recall, F1
        precision = len(true_positives) / len(predicted_files_set) if predicted_files_set else 0
        recall = len(true_positives) / len(ground_truth_files) if ground_truth_files else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Top N precision
        top_5_precision = len(top_5_true_positives) / len(top_5_files) if top_5_files else 0
        top_10_precision = len(top_10_true_positives) / len(top_10_files) if top_10_files else 0
        
        # Print comparison results
        print(f"\n=== Comparison with Ground Truth ===")
        print(f"Total Files in Ground Truth: {len(ground_truth_files)}")
        print(f"Total Files Predicted as Buggy: {len(predicted_files_set)}")
        
        # Print metrics
        print(f"\n=== Model Performance ===")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1_score:.2f}")
        print(f"Top 5 Precision: {top_5_precision:.2f}")
        print(f"Top 10 Precision: {top_10_precision:.2f}")
        
        # Print mismatched files
        if len(false_positives) > 0:
            print(f"\nFalse Positives (Files predicted as buggy but not in ground truth):")
            for fp in false_positives:
                print(f"  - {fp}")
        
        if len(false_negatives) > 0:
            print(f"\nFalse Negatives (Files in ground truth but not predicted):")
            for fn in false_negatives:
                print(f"  - {fn}")
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "top_5_precision": top_5_precision,
            "top_10_precision": top_10_precision,
            "true_positives": len(true_positives),
            "false_positives": len(false_positives),
            "false_negatives": len(false_negatives),
            "ground_truth_count": len(ground_truth_files),
            "predicted_buggy_count": len(predicted_files_set)
        }
    except Exception as e:
        logger.error(f"Error comparing with ground truth: {e}")
        return {}

def print_predictions(predictions):
    """
    Print top predicted buggy files with confidence scores.
    
    Args:
        predictions (list): List of dictionaries with predictions
    """
    print("\n=== Top 10 Most Likely Buggy Files ===")
    
    for i, prediction in enumerate(predictions[:10]):
        print(f"{i+1}. {prediction['file_path']} - Confidence: {prediction['confidence']:.4f}")
    
    # Print summary counts
    buggy_count = sum(1 for p in predictions if p['is_buggy'])
    print(f"\nTotal files analyzed: {len(predictions)}")
    print(f"Files predicted as buggy: {buggy_count}")
    print(f"Files predicted as clean: {len(predictions) - buggy_count}")

def main():
    """Main function to run the ML defect predictor."""
    parser = argparse.ArgumentParser(description='ML-based Defect Predictor')
    parser.add_argument('--config', default='config.json', help='Path to config file')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Generate predictions')
    parser.add_argument('--save-model', action='store_true', help='Save the trained model')
    parser.add_argument('--load-model', action='store_true', help='Load a pre-trained model')
    parser.add_argument('--model-path', default='ml_defect_model.joblib', help='Path to model file')
    parser.add_argument('--ground-truth', default='ground_truth.csv', help='Path to ground truth file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Initialize predictor
    predictor = MLDefectPredictor(config)
    
    # Load model if requested
    if args.load_model:
        if predictor.load_model(args.model_path):
            logger.info(f"Loaded model from {args.model_path}")
        else:
            logger.error(f"Failed to load model from {args.model_path}")
            return
    
    # Train model if requested
    if args.train:
        logger.info("Training model...")
        metrics = predictor.train()
        
        if "error" in metrics:
            logger.error(f"Training failed: {metrics['error']}")
            return
        
        print("\n=== Training Metrics ===")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Buggy files in test set: {metrics['buggy_files_count']} out of {metrics['total_files_count']}")
        
        print("\n=== Feature Importances ===")
        for feature, importance in sorted(metrics['feature_importances'].items(), key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance:.4f}")
        
        # Save model if requested
        if args.save_model:
            if predictor.save_model(args.model_path):
                logger.info(f"Model saved to {args.model_path}")
            else:
                logger.error("Failed to save model")
    
    # Generate predictions if requested
    if args.predict:
        logger.info("Generating predictions...")
        predictions = predictor.predict()
        
        if not predictions:
            logger.error("No predictions generated")
            return
        
        # Print top predictions
        print_predictions(predictions)
        
        # Compare with ground truth if file exists
        if os.path.exists(args.ground_truth):
            compare_with_ground_truth(predictions, args.ground_truth)
        else:
            logger.warning(f"Ground truth file {args.ground_truth} not found. Skipping comparison.")

if __name__ == "__main__":
    main() 