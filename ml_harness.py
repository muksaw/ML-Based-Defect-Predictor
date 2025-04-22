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
        predicted_files (list): List of dictionaries with predicted buggy files (each dictionary contains 'file_path' and 'is_buggy').
        ground_truth_file (str): Path to the ground truth CSV file.
        config_url (str): The URL to filter the ground truth by.
        config_start_date (str): The start date (YYYY-MM-DD) to filter the ground truth by.
        config_end_date (str): The end date (YYYY-MM-DD) to filter the ground truth by.
    
    Returns:
        dict: Comparison metrics including precision, recall, F1 score, top 5/10 precision, and counts of true/false positives/negatives.
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

        # Get top N files by confidence (also use only filenames)
        top_5_files = {os.path.basename(f['file_path']) for f in predicted_files[:5]} if len(predicted_files) >= 5 else predicted_files_set
        top_10_files = {os.path.basename(f['file_path']) for f in predicted_files[:10]} if len(predicted_files) >= 10 else predicted_files_set

        # Calculate true positives, false positives, and false negatives
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
            print(list(false_positives)[:20])  # Truncating the list for print
        
        if len(false_negatives) > 0:
            print(f"\nFalse Negatives (Files in ground truth but not predicted):")
            print(list(false_negatives)[:20])  # Truncating the list for print
        
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
        # Add relative risk to the output with category
        risk_info = f" - Risk: {prediction.get('relative_risk', 0):.2f} ({prediction.get('risk_category', 'Unknown')})" if 'relative_risk' in prediction else ""
        print(f"{i+1}. {prediction['file_path']} - Confidence: {prediction['confidence']:.4f}{risk_info}")
    
    # Print summary counts
    buggy_count = sum(1 for p in predictions if p['is_buggy'])
    print(f"\nTotal files analyzed: {len(predictions)}")
    print(f"Files predicted as buggy: {buggy_count}")
    print(f"Files predicted as clean: {len(predictions) - buggy_count}")
    
    # Add a risk breakdown if we have predictions
    if len(predictions) > 0 and 'risk_category' in predictions[0]:
        risk_counts = {}
        for p in predictions:
            category = p.get('risk_category', 'Unknown')
            risk_counts[category] = risk_counts.get(category, 0) + 1
        
        print("\n=== Risk Category Breakdown ===")
        for category, count in sorted(risk_counts.items(), key=lambda x: ['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High', 'Unknown'].index(x[0])):
            print(f"{category} Risk: {count} files ({count/len(predictions)*100:.1f}%)")
    
    # Get time span and threshold information
    time_span = None
    adjusted_threshold = None
    original_threshold = None
    
    if len(predictions) > 0:
        if '_time_span' in predictions[0]:
            time_span = predictions[0].get('_time_span', 0)
        if '_adjusted_threshold' in predictions[0]:
            adjusted_threshold = predictions[0].get('_adjusted_threshold', 0.7)
        if '_original_threshold' in predictions[0]:
            original_threshold = predictions[0].get('_original_threshold', 0.7)
    
    # Explain metrics and thresholds
    print("\n=== Understanding the Metrics ===")
    
    # Explain confidence scores
    print("\n* Confidence Score (0-1):")
    print("  This is the machine learning model's prediction confidence that a file contains bugs.")
    print("  - 0.5-0.7: Low confidence - the file might contain bugs")
    print("  - 0.7-0.85: Medium confidence - the file likely contains bugs")
    print("  - 0.85-1.0: High confidence - the file very likely contains bugs")
    
    # Explain the threshold adjustment if applicable
    if time_span and adjusted_threshold and original_threshold:
        print(f"\n* Confidence Threshold Adjustment:")
        print(f"  Base threshold: {original_threshold:.2f}")
        print(f"  Adjusted threshold: {adjusted_threshold:.2f}")
        print(f"  This adjustment accounts for the {time_span} day analysis period.")
        print("  - Longer time periods use higher thresholds to reduce false positives")
        print("  - Only files with confidence above this threshold are shown")
    
    # Explain relative risk score
    print("\n* Relative Risk Score:")
    print("  This normalized metric compares each file's risk against repository averages.")
    print("  Relative Risk considers bug history, commit patterns, and code complexity.")
    print("  - <0.5: Low Risk - significantly safer than repository average")
    print("  - 0.5-1.0: Medium-Low Risk - somewhat safer than repository average")
    print("  - 1.0-1.5: Medium Risk - around repository average")
    print("  - 1.5-3.0: Medium-High Risk - higher risk than repository average")
    print("  - >3.0: High Risk - significantly higher risk than repository average")

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
    # Add time decay factor parameter
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
        print(f"ML-Based Defect Predictor - Analysis started at {timestamp}")
        print(f"\n=== Configuration ===")
        print(f"Repository: {config['url_to_repo']}")
        print(f"Branch: {config['branch']}")
        print(f"Date range: {config.get('from_date', 'N/A')} to {config.get('to_date', 'N/A')}")
        print(f"Max commits: {config.get('max_commits', 'All')}")
        print(f"File extensions: {', '.join(config.get('file_extensions', ['.py']))}")
        print(f"Base confidence threshold: {config.get('confidence_threshold', 0.7):.2f}")
        print(f"Time decay factor: {config.get('time_decay_factor', 30)} days")
        
        # Load model if requested
        if args.load_model:
            if predictor.load_model(args.model_path):
                logger.info(f"Loaded model from {args.model_path}")
                print(f"\nLoaded pre-trained model from {args.model_path}")
            else:
                logger.error(f"Failed to load model from {args.model_path}")
                print(f"\nError: Failed to load model from {args.model_path}")
                return
        
        # Train model if requested
        if args.train:
            logger.info("Training model...")
            print("\n=== Starting Model Training ===")
            print("Training using enhanced metrics including time-weighted analysis and relative risk scoring...")
            max_commits = config.get('max_commits', 6000)
            metrics = predictor.train()
            
            if "error" in metrics:
                logger.error(f"Training failed: {metrics['error']}")
                print(f"\nTraining failed: {metrics['error']}")
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
                    print(f"\nModel saved to {args.model_path}")
                else:
                    logger.error("Failed to save model")
                    print("\nError: Failed to save model")
        
        # Generate predictions if requested
        if args.predict:
            logger.info("Generating predictions...")
            print("\n=== Generating Predictions ===")
            print("Using enhanced defect prediction with adaptive confidence threshold and time-weighted analysis...")
            max_commits = config.get('max_commits', 6000)
            predictions = predictor.predict()
            
            if not predictions:
                logger.error("No predictions generated")
                print("\nError: No predictions generated")
                return
            
            # Print top predictions
            print_predictions(predictions)
            
            # Inform user about feature table export
            print("\n=== Feature Table Export ===")
            print("A CSV file containing all extracted features has been saved to the outputs directory.")
            print("This table includes metrics such as:")
            print("  - Number of commits")
            print("  - Lines added/deleted")
            print("  - Code complexity")
            print("  - Bug fix history")
            print("  - Time-based metrics")
            print("You can use this data for further analysis or with other ML tools.")
            
            # Compare with ground truth if file exists
            if os.path.exists(args.ground_truth):
                url = config['url_to_repo']
                fromDate = config['from_date']
                toDate = config['to_date']
                compare_with_ground_truth(predictions, args.ground_truth, url, fromDate, toDate)
            else:
                logger.warning(f"Ground truth file {args.ground_truth} not found. Skipping comparison.")
                print(f"\nNote: Ground truth file {args.ground_truth} not found. Skipping comparison.")

if __name__ == "__main__":
    main() 