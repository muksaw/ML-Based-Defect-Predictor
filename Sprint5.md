# Sprint 5 Summary: ML-Based Defect Predictor

## Accomplishments in Sprint 5

1. **Feature Table Export Implementation**:
   - Added CSV export functionality for feature tables as requested by professor
   - Enhanced filename format for better organization and tracking
   - Implemented descriptive naming convention with timestamps and analysis types
   - Enabled external validation and analysis of prediction metrics

2. **ML Implementation Refinement**:
   - Evaluated AutoGluon for potential ML delegation
   - Determined current implementation superiority due to:
     - Already far along in current implementation
     - Optimized performance for defect prediction
     - Custom features specifically designed for defect analysis

3. **Software Testing Concept Integration**:
   - Enhanced existing testing concepts implementation
   - Added new testing methodologies from course material
   - Introduced more logic coverage for bug detection

## Technical Improvements

1. **CSV Export System**:
   ```python
   # Enhanced file naming for better organization
   timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
   analysis_type = 'historical' if historical_data else 'prediction'
   csv_path = os.path.join(output_dir, 
                          f'feature_table_{analysis_type}_{timestamp}.csv')
   df.to_csv(csv_path, index=False)
   ```

2. **Enhanced Logic Coverage**:
   ```python
   def is_bug_fix(self, commit_msg):
       return any(keyword in commit_msg.lower() for keyword in [
           'fix', 'bug', 'issue', 'error', 'crash', 'problem',
           'defect', 'fault', 'flaw', 'incorrect', 'regression'
       ])
   ```

3. **Output Organization**:
   ```python
   # More descriptive output filename with timestamp and run configuration
   config_info = []
   if args.train:
       config_info.append('train')
   if args.predict:
       config_info.append('predict')
   if args.load_model:
       config_info.append('loaded_model')
   
   run_type = '_'.join(config_info) if config_info else 'analysis'
   output_file = os.path.join(OUTPUTS_DIR, 
                             f"{output_base}_{run_type}_{timestamp}.txt")
   ```

## Integration of Course Concepts

1. **Test Automation (Theme 1)**:
   - Automated execution through command-line interface:
     ```python
     parser.add_argument('--train', action='store_true', help='Train the model')
     parser.add_argument('--predict', action='store_true', help='Generate predictions')
     parser.add_argument('--save-model', action='store_true', help='Save the trained model')
     ```
   - Configurable parameters for different test scenarios
   - Continuous output logging and result tracking

2. **Test Design (Theme 2)**:
   - Systematic approach to feature extraction
   - Well-defined metrics for risk assessment:
     ```python
     def calculate_relative_risk(self, file_metrics):
         # Systematic risk assessment using multiple metrics
         commit_counts = [m['n_commits'] for m in file_metrics.values()]
         bug_fix_counts = [m['bug_fix_count'] for m in file_metrics.values()]
     ```

3. **Logic Coverage**:
   - Enhanced bug detection logic with comprehensive criteria
   - Improved pattern matching for defect identification
   - Systematic approach to categorizing code changes

## Challenges Addressed

1. **Output Organization**:
   - Previous: Output files lacked clear identification
   - Solution: Implemented descriptive naming with timestamps and analysis types
   - Result: Better traceability and organization of analysis results

2. **ML Framework Decision**:
   - Challenge: Evaluate potential switch to AutoGluon
   - Analysis: Found significant performance overhead
   - Decision: Maintained current implementation for better efficiency
   - Result: Preserved optimized performance while meeting course objectives
   
3. **Refinement of feature selection and clarity regarding our features**
   - See next section.

## Feature Selection output, and how it works
The model uses 12 key features that are automatically extracted from git history. [Click here](https://docs.google.com/spreadsheets/d/1s1Br2VPYxl9xDe6LOxGbnZ0dsy_BaZuD49s9IsvwKVs/edit?usp=sharing)
### Most Important Features (based on feature importance scores)
- `n_authors` (21.33%)
- `weighted_bugs` (21.27%)
- `bug_fix_count` (15%)
- `relative_risk` (12.08%)
- `n_commits` (10.31%)

### Time-Weighted Features
- `weighted_commits`: Commits are weighted based on recency using exponential decay
  - More recent commits have higher weights
  - Uses a 30-day decay factor (configurable)
  - Helps identify files with recent activity

### Code Change Metrics
- `n_lines_added`: Total lines added to a file
- `n_lines_deleted`: Total lines deleted from a file
- These help identify files with high churn, which often correlates with bugginess

### Complexity Metrics
- `avg_complexity`: Average cyclomatic complexity of functions in the file
  - Higher complexity = more decision points = higher chance of bugs
  - Measured using standard cyclomatic complexity metrics

### Time-Based Features
- `age_days`: How old the file is (days since first commit)
- `recent_modified_days`: Days since last modification
- Older files with recent changes often indicate maintenance issues

### Bug-Related Features
- `bug_fix_count`: Number of commits marked as bug fixes
- `weighted_bugs`: Time-weighted bug fix count
- Bug fixes are identified by commit messages containing keywords like "fix", "bug", "issue", etc.

### Commit Patterns
- `commit_density`: Commits per month (normalized by file age)
- High density often indicates unstable or frequently changing code

### Risk Assessment
- `relative_risk`: Normalized risk score comparing file metrics to repository averages
- `risk_category`: Categorized risk levels (Low, Medium-Low, Medium-High, High)
- Based on multiple factors including bug history, complexity, and change patterns

### Buggy Determination
- `is_buggy`: Binary label (0/1) indicating if a file has had bug fixes
- Determined by analyzing commit history for bug-fix related commits
- Used as the target variable for training the model
## Plans for Future Enhancements

**Regression Testing Integration**:
   - Implement incremental analysis capabilities
   - Focus analysis on recently modified files
   - Optimize performance for large codebases
   - Add change-based prediction refinement

## Conclusion

Sprint 5 has successfully integrated core software testing concepts while maintaining the project's practical utility. The addition of CSV export functionality and enhanced logic coverage demonstrates our application of course concepts, while the decision to maintain our current ML implementation shows pragmatic engineering judgment.

The improvements in output organization and feature extraction continue to enhance the tool's usability, while the integration of software testing principles strengthens its theoretical foundation. Moving forward, the planned implementations of graph coverage and regression testing concepts will further align the project with course objectives while maintaining its practical value for defect prediction.

Our enhanced model now provides:
- Better organized and more traceable outputs
- Stronger integration with software testing principles
- Maintained high performance and accuracy
- Clear path for future enhancements

These improvements position the tool as both a practical defect prediction system and a demonstration of core software testing concepts. 
