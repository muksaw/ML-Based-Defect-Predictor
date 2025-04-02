# Defect Predictor using RepoMiner

This project provides an implementation of a Defect Predictor by leveraging RepoMiner and PyDriller to analyze commit history and predict the risk associated with files in a given revision. It extends the RepoMiner example and includes risk calculation based on commit history.

## Key Features

- **Machine Learning-Based Defect Prediction**: Uses Random Forest classifier to identify potentially buggy files
- **Time-Weighted Analysis**: Gives more weight to recent commits and bug fixes
- **Relative Risk Scoring**: Compares each file against repository averages
- **Adaptive Confidence Threshold**: Automatically adjusts prediction sensitivity based on time span
- **Risk Categorization**: Categorizes files into risk levels (Low to High)
- **Test File Exclusion**: Excludes test files from analysis

# Setup Instructions


## Installation
Follow the steps below to set up and run the project after cloning its repo

### 2. Build the docker file
Create a docker image (see Dockerfile) for the project: 
```bash
docker build -t defect-predictor .
```

### 3. Run the project
Run the docker container from the image and mount the outputs directory to save results:
```bash
docker run --rm -it -v $(pwd)/outputs:/app/outputs defect-predictor
```

  
### 4. Login and run 
Create a docker container and run a bash shell in it. From there, you can modify the file config.json as you wish.
```bash
 docker run -it --rm defect-predictor bash
```

## Understanding Risk Scores

The model uses two primary metrics to evaluate file risk:

### Confidence Score (0-1)
The machine learning model's confidence that a file contains bugs:
- 0.5-0.7: Low confidence
- 0.7-0.85: Medium confidence
- 0.85-1.0: High confidence

### Relative Risk Score
A normalized metric that compares each file's risk against repository averages:
- <0.5: Low Risk - significantly safer than average
- 0.5-1.0: Medium-Low Risk - somewhat safer than average
- 1.0-1.5: Medium Risk - around average
- 1.5-3.0: Medium-High Risk - higher risk than average
- >3.0: High Risk - significantly higher risk than average

## Configuration Options

The `config.json` file allows you to customize:

```json
{
    "url_to_repo": "Repository URL to analyze",
    "clone_repo_to": "Local path to clone the repository",
    "branch": "Branch to analyze",
    "from_date": "Start date for analysis (YYYY-MM-DD)",
    "to_date": "End date for analysis (YYYY-MM-DD)",
    "confidence_threshold": "Base confidence threshold (0.0-1.0)",
    "model_path": "Path to save/load the trained model",
    "file_extensions": ["Extensions to include in analysis"],
    "max_commits": "Maximum number of commits to analyze",
    "time_decay_factor": "Half-life for time weighting in days"
}
```

## Overview of files

  ### ml_defect_predictor.py:
  - Main defect prediction algorithm with machine learning and risk scoring
  
  ### ml_harness.py
  - Script to run the defect predictor and display results
  
  ### requirements.txt
  - Lists all the dependencies and libraries required to run the project seamlessly.
  
  ### ground_truth.csv
  - Provides a reference dataset containing start and end dates along with the modified files. It is used to compare expected results with actual outputs.
  
  ### config.json
  - Allows you to specify the repository to be tested, the local path where it should be cloned, the branch to analyze, and the start and end dates.
  
  ### Docker 
  - Sets up the necessary environment with Python, Git, dependencies, and your project files to run the defect predictor seamlessly inside a container.
