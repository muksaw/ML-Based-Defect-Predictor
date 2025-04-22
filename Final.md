# Final Sprint Summary: ML-Based Defect Predictor

## Accomplishments in Final Sprint

1. **Configuration Finalization**:
   - Created config.json for each repo so that we can readily paste it in one at a time.

## Project Evaluation

1. **Accuracy and Performance**:
   - The model demonstrated **strong accuracy**, consistently identifying the majority of known buggy files
   - Relative risk scoring and time-weighted features added useful context for prioritizing defect review

2. **Precision Challenges**:
   - Overall **precision remained lower than ideal**, primarily due to:
     - Limited size of the `ground_truth.csv` compared to the scale of some repositories
     - Large commit volumes that introduced noise and increased the number of flagged files
   - Despite these challenges, risk-based prioritization mitigated much of the noise and enabled practical application

## Conclusion

With all sprints complete and the system fully operational, the ML-Based Defect Predictor is ready for demonstration. The tool now integrates configurable repository settings, interpretable risk scoring, adaptive thresholding, and comprehensive output organization. While the imbalance between commit data and ground truth labels impacted precision, the tool remains highly useful for defect prioritization and ongoing code quality analysis. The project represents a strong blend of practical utility and core software testing principles.
