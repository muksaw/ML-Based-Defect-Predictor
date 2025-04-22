#!/usr/bin/env python3
import os
import json
import pandas as pd
import argparse
import sys
from ml_defect_predictor import MLDefectPredictor
import logging
from datetime import datetime
from contextlib import contextmanager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create outputs directory if it doesn't exist
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

@contextmanager
def redirect_stdout(output_file=None):
    """
    Context manager to redirect stdout to a file while still displaying on console.
    
    Args:
        output_file (str): Path to output file
    """
    original_stdout = sys.stdout
    
    if output_file:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        file_stdout = open(output_file, 'w')
        
        class DualOutput:
            def write(self, message):
                file_stdout.write(message)
                original_stdout.write(message)
                
            def flush(self):
                file_stdout.flush()
                original_stdout.flush()
        
        sys.stdout = DualOutput()
    
    try:
        yield
    finally:
        if output_file:
            sys.stdout = original_stdout
            file_stdout.close()
            logger.info(f"Full output saved to {output_file}")

def load_config(config_file):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)

def truncate_list(items, max_display=10):
    """
    Truncate a list for display purposes.
    
    Args:
        items (list): List to truncate
        max_display (int): Maximum number of items to display
        
    Returns:
        str: Truncated list as string
    """
    if len(items) <= max_display:
        return "\n".join(f"  - {item}" for item in items)
    
    first_half = items[:max_display//2]
    last_half = items[-(max_display//2):]
    
    truncated = "\n".join(f"  - {item}" for item in first_half)
    truncated += f"\n  ... {len(items) - max_display} more files (see output.txt for complete list) ...\n"
    truncated += "\n".join(f"  - {item}" for item in last_half)
    
    return truncated


def compare_with_ground_truth(predicted_files, ground_truth_file, config_url, config_start_date, config_end_date):
    """
    Compares the output of defect predictor model with the ground truth CSV file filtered by URL and dates.
    
    Args:
        predicted_files (list): List of dictionaries with predicted buggy files.
        ground_truth_file (str): Path to the ground truth CSV file.
        config_url (str): The URL to filter the ground truth by.
        config_start_date (str): The start date (YYYY-MM-DD) to filter the ground truth by.
        config_end_date (str): The end date (YYYY-MM-DD) to filter the ground truth by.
    """
    try:
        # Load ground truth
        df = pd.read_csv(ground_truth_file)
        
        # Convert date columns in ground truth to datetime (YYYY-MM-DD format)
        df['from_date'] = pd.to_datetime(df['from_date'], format='%Y-%m-%d')
        df['to_date'] = pd.to_datetime(df['to_date'], format='%Y-%m-%d')
        
        # Convert config.json dates to datetime (YYYY-MM-DD)
        config_from = pd.to_datetime(config_start_date, format='%Y-%m-%d')
        config_to = pd.to_datetime(config_end_date, format='%Y-%m-%d')
        
        # Filter ground truth entries by URL and date range
        mask = (
            (df['github_url'] == config_url) &
            (config_from >= df['from_date']) &
            (config_to <= df['to_date'])
        )
        df_filtered = df[mask]
        
        # Collect all modified files from filtered entries
        ground_truth_files = set()
        for file_list in df_filtered['risky_files']:
            files = [os.path.basename(f.strip()) for f in file_list.split(',')]
            ground_truth_files.update(files)
        
        # Get predicted files (those with is_buggy=True) — use only filenames
        predicted_files_set = {os.path.basename(f['file_path']) for f in predicted_files if f['is_buggy']}
        
        # Calculate true positives, false positives, and false negatives
        true_positives = predicted_files_set.intersection(ground_truth_files)
        false_positives = predicted_files_set - ground_truth_files
        false_negatives = ground_truth_files - predicted_files_set
        
        # Print mismatched files in the original format
        print("\nFalse Positives (Files predicted but not in ground truth):")
        for fp in false_positives:
            print(fp)
        
        print("\nFalse Negatives (Files in ground truth but not predicted):")
        for fn in false_negatives:
            print(fn)
        
        # Calculate and print accuracy
        accuracy = len(true_positives) / (len(true_positives) + len(false_positives) + len(false_negatives)) if (len(true_positives) + len(false_positives) + len(false_negatives)) > 0 else 0
        print(f"\nAccuracy: {accuracy:.2f}")
        
        # Return metrics dictionary for file output
        return {
            "accuracy": accuracy,
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
    Print predictions in the format matching the original harness.
    
    Args:
        predictions (list): List of dictionaries with predictions
    """
    # Get all predicted buggy files as a set
    buggy_files = {os.path.basename(p['file_path']) for p in predictions if p['is_buggy']}
    print("Risky Python files:", buggy_files)
    
    # Print risk scores
    print("\nCalculating risk scores...")
    print("\nRisk Scores (File Name -> Risk Value):")
    # Create a list of tuples (filename, risk) for sorting
    risk_scores = [(os.path.basename(pred['file_path']), pred['relative_risk']) 
                  for pred in predictions if pred['is_buggy']]
    # Sort by risk score in descending order
    risk_scores.sort(key=lambda x: x[1], reverse=True)
    # Print sorted scores
    for filename, risk in risk_scores:
        print(f"{filename} -> {risk:.2f}")
    
    # Print top 3 risky files
    print("\nTop 3 Risky Files:")
    for filename, risk in risk_scores[:3]:
        print(f"  - {filename} -> {risk:.2f}")

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
    parser.add_argument('--max-commits', type=int, help='Maximum number of commits to analyze')
    parser.add_argument('--output-file', help='File to save detailed output')
    parser.add_argument('--time-decay', type=float, help='Time decay factor for commit weighting (in days)')
    
    args = parser.parse_args()
    
    # Create more descriptive output filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_file:
        output_base = os.path.splitext(args.output_file)[0]
    else:
        output_base = 'defect_prediction'
        
    # Add run configuration to filename
    config_info = []
    if args.train:
        config_info.append('train')
    if args.predict:
        config_info.append('predict')
    if args.load_model:
        config_info.append('loaded_model')
        
    # Create descriptive filename
    run_type = '_'.join(config_info) if config_info else 'analysis'
    output_file = os.path.join(OUTPUTS_DIR, f"{output_base}_{run_type}_{timestamp}.txt")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Override max_commits if provided via command line
    if args.max_commits:
        config['max_commits'] = args.max_commits
    
    # Override time decay factor if provided
    if args.time_decay:
        config['time_decay_factor'] = args.time_decay
    
    # Initialize predictor
    predictor = MLDefectPredictor(config)
    
    # Redirect output to both console and file
    with redirect_stdout(output_file):
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

                print("\nNo risk scores calculated.")
                return
            
            if predictions:
                # Print predictions in original harness format
                print_predictions(predictions)
            
                # Compare with ground truth if file exists
                if os.path.exists(args.ground_truth):
                    print("\nComparing with Ground Truth...")
                    compare_with_ground_truth(
                        predictions,
                        args.ground_truth,
                        config['url_to_repo'],
                        config['from_date'],
                        config['to_date']
                    )
                else:
                    logger.warning(f"Ground truth file {args.ground_truth} not found. Skipping comparison.")

if __name__ == "__main__":
    main() 