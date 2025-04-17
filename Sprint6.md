# Sprint 6 Summary: ML-Based Defect Predictor

## Accomplishments in Sprint 6

1. **Enhanced Model Explanation and Feature Analysis**:
   - Improved transparency around feature selection and importance
   - Documented feature extraction 
   - Analyze and explain our prediction outputs and risk scoring

2. **Model Learning and Decision Process Documentation**:
   - Detailed explanation of what the model is learning from
   - Clarified label determination and ground truth methodology
   - Improved understanding of confidence scores and their interpretation

3. **Risk Assessment Formulation**:
   - Documented detailed mathematical calculations for complexity metrics
   - Explained threshold adaptation based on repository characteristics
   - Provided risk categorization meaning and decision boundaries

## Technical Deep Dive

### What the Model is Learning

The Random Forest classifier learns to identify potentially buggy files by analyzing patterns from 12 key features:

```
file_path,n_commits,weighted_commits,n_authors,n_lines_added,n_lines_deleted,avg_complexity,
age_days,recent_modified_days,bug_fix_count,weighted_bugs,commit_density,relative_risk,
risk_category,is_buggy
```

- Primary signal comes from repository history and code structure patterns
- Model learns which combination of features correlates with bug occurrence
- Decision trees split on features in order of their predictive power
- Most influential features from our test runs:
  - Number of authors (21.33%)
  - Time-weighted bug fixes (21.27%)
  - Raw bug fix count (15.00%)
  - Relative risk score (12.08%)
  - Number of commits (10.31%)

### Feature Extraction Sources

Features are derived from multiple sources:

1. **Git Repository Mining** (via PyDriller):
   - Commit history (counts, authors, timestamps)
   - Code changes (lines added/deleted)
   - Complexity metrics (cyclomatic complexity)
   
2. **Derived Calculations**:
   - Time-weighted metrics using exponential decay
   - Commit density normalized by file age
   - Relative metrics compared to repository averages

3. **Label Determination**:
   - `is_buggy` label is automatically determined from commit history
   - Uses keyword detection in commit messages (fix, bug, issue, etc.)
   - Label is binary (0/1) indicating if a file has been part of a bug fix
   - Acts as ground truth for model training

### Mathematical Formulation of Risk

1. **Weighted Bug Density Calculation**:
   ```
   weighted_bug_density = bug_ratio * (1 + 0.5 * (complexity_factor - 1))
   ```
   
   Where:
   - `bug_ratio = bug_fix_count / n_commits`
   - `complexity_factor = file_complexity / avg_repo_complexity`
   - `0.5` is the weight coefficient for complexity influence

2. **Relative Risk Calculation**:
   ```
   relative_risk = weighted_bug_density * (1 + 0.3 * (commit_factor - 1))
   ```
   
   Where:
   - `commit_factor = file_commits / avg_repo_commits`
   - `0.3` is the weight coefficient for commit frequency influence

These formulas create a normalized risk score where generally speaking:
- 1.0 represents average repository risk
- Values >1.0 indicate higher than average risk
- Values <1.0 indicate lower than average risk

### Prediction Outputs and Thresholding

The model produces several key outputs:

1. **Confidence Score** (0-1):
   - Probability that a file contains bugs
   - Higher values indicate higher likelihood
      - Used to determine final prediction

2. **Adaptive Threshold**:
   ```
   adjusted_threshold = base_threshold + min(0.2, (time_span / 365) * 0.1)
   ```
   
   - Base threshold: 0.70
   - Adjusts automatically for longer analysis periods

3. **Risk Categorization**:
   - Low: < 0.5 (significantly safer than average)
   - Medium-Low: 0.5-1.0 (somewhat safer than average)
   - Medium: 1.0-1.5 (average risk)
   - Medium-High: 1.5-3.0 (higher than average risk)
   - High: > 3.0 (significantly higher risk)

## Integration of Course Concepts

1. **Test Oracle Principles**:
   - Used past bug fixes as oracle for model validation
   - Implemented precision and recall metrics for evaluation
   - Compared predictions against known ground truth

2. **Decision Table Testing**:
   - Evaluated model performance across different thresholds
   - Implemented multiple decision paths in risk categorization


## Future Enhancements

1. **Model Performance Metrics Enhancement**:
   - Add more sophisticated evaluation metrics
   - Create visualization of feature importance and relationship

2. **Continuous Feedback Loop**:
   - Develop mechanism to incorporate developer feedback
   - Implement verification of predictions against actual bugs
   - Create incremental learning pipeline for model improvement
