
# Presentation Points: ML-Based Defect Predictor

## What We've Implemented

1. **Machine Learning Approach**: 
   - Implemented a Random Forest classifier instead of the simple time-decay risk function in the original code
   - Uses historical bug patterns to make predictions rather than just recency of changes

2. **Feature Engineering**:
   - Goal is to extract 8 distinct metrics from repository history (commits, authors, code complexity, etc.)
   - Standardize features for better model performance
   - Handle timezone-aware datetime comparison

3. **Modern ML Pipeline**:
   - Train/test split for proper evaluation
   - Feature scaling for numerical stability
   - Model persistence (save/load functionality)
   - Confidence scores for prediction ranking

4. **Enhanced Evaluation Framework**:
   - Precision, recall, and F1 metrics
   - Feature importance analysis
   - Top-N accuracy metrics (Top-5, Top-10)

## How the code works

# MLDefectPredictor

A class for extracting code‑change metrics from a Git repo, training a defect‑prediction model, and predicting which files are likely to contain bugs.

## Methods

### __init__(config)
Initialize with repository URL, local clone path, branch, and optional date filters.

### extract_features(historical_data=True)
Collect file‑level metrics from the repo’s commit history; optionally label files as “buggy” for training.

### prepare_data(df, for_training=True)
Select, clean, and scale features; split into train/test sets if training, otherwise prepare data for prediction.

### train_model(X_train, y_train)
Fit a Random Forest classifier on the provided training data.

### train()
Run the full pipeline extract features, prepare data, train the model, evaluate performance and return evaluation metrics.

### predict()
Extract current repository features and use the trained model to predict & rank potentially buggy files.

### save_model(path="ml_defect_model.joblib")
Serialize the trained model and scaler to disk for later reuse.

### load_model(path="ml_defect_model.joblib")
Load a previously saved model and scaler into memory for inference.

## Plans for Next Sprint

1. **Data Enhancement**:
   - Use real historical repositories with documented bugs
   - Implement synthetic bug pattern generation

2. **Model Improvements**:
   - Implement class weighting for imbalanced data

3. **Feature Expansion**:
   - Integrate static code analysis metrics

4. **Usability**:
   - (Eventually) Add visualization of prediction results
   - Implement incremental training capability

