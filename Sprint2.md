
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

1. **Feature Extraction**:
   ```python
   def extract_features(self, historical_data=True):
       # Traverses repository commits
       # Collects metrics for each file
       # Labels files from bug-fixing commits if training
   ```

2. **Training Process**:
   ```python
   def train(self):
       # Extract features from historical data
       # Split data into training and test sets
       # Train Random Forest model
       # Evaluate and return performance metrics
   ```

3. **Prediction**:
   ```python
   def predict(self):
       # Extract features from current repository state
       # Apply trained model to predict buggy files
       # Return sorted list by confidence score
   ```

4. **Key Innovations**:
   - Bug detection using keywords in commit messages
   - Timezone-aware comparisons for cross-repository compatibility

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

