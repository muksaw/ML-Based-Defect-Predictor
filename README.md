# Defect Predictor using RepoMiner

This project provides an implementation of a Defect Predictor by leveraging RepoMiner and PyDriller to analyze commit history and predict the risk associated with files in a given revision. It extends the RepoMiner example and includes risk calculation based on commit history.

## Features
- Traverse Git repositories to identify modified Python files.
- Classify fixing commits using a `FixingCommitClassifier`.
- Calculate risk scores for Python files based on bug revisions and modification frequency.

# Setup Instructions


## Installation
Follow the steps below to set up and run the project:

### 1. Clone the repository

```bash
git clone https://github.ncsu.edu/mmyaka/DefectPredictorTemplate.git
```
- Change the current working directory 
```bash
cd DefectPredictorTemplate
```

### 2. Build the docker file
Create a docker image (see Dockerfile) for the project: 
```bash
docker build -t defect-predictor .
```

### 3. Run the project
Run the docker container from the image. See instruction RUN at the end of Dockerfile. It will execute the harness function within a docker container, print the results, and exit.
```bash
docker run --rm -it defect-predictor
```
  #### Expected output:
  ```plaintext
  Retrieving modified Python files...
  Modified files debug output: {'word_frequency.py'}
  
  Comparing with Ground Truth...
  Total Files in Ground Truth: 1
  Total Files Predicted by the defect predictor: 1
  
  False Positives (Files predicted but not in ground truth):
  word_frequency.py
  
  False Negatives (Files in ground truth but not predicted):
  factorial.py
  
  Precision: 0.00
  Recall: 0.00
  F1 Score: 0.00
  
  Calculating risk scores...
  
  Risk Scores (File Name -> Risk Value):
  word_frequency.py -> 2.0
  ```

  ### Explanation of outputs:
  - **Modified Files**: Lists all Python files that were modified during the specified period, based on the defect predictor analysis.
  - **Comparison with Ground Truth**: 
    The comparison is based on a ground truth file, which contains a list of files modified between a start and end date. This ground truth data is compared with the predictions made by the sample defect predictor model to assess its accuracy.
    - **False Positives**: Files that were predicted by the model but are not in the ground truth.
    - **False Negatives**: Files that are in the ground truth but were not predicted by the model.
    - **Precision, Recall, and F1 Score**: Metrics that summarize the prediction accuracy of the model.
  - **Risk Scores**: Displays the calculated risk scores for each modified file.
  
  
### 4. Login and run 
Create a docker container and run a bash shell in it. From there, you can modify the file config.json as you wish.
```bash
 docker run -it --rm defect-predictor bash
```

## Overview of files

  ### SampleDefectPredictor.py:
  - This script identifies and lists Python files modified during a specified time frame in a Git repository. Additionally, it calculates a risk score for each file, which can help in prioritizing potential defect analysis. You can customize this script to implement your unique defect prediction model and risk assessment logic.
  ### harness.py
  - This script runs the defect predictor (SampleDefectPredictor.py) logic to generate a list of modified files, calculate their risk scores, and compare the results with a ground truth file. Please note that this file should not be modified, as your implementation is expected to function using it as is.
  ### requirements.txt
  - Lists all the dependencies and libraries required to run the project seamlessly.
  ### ground_truth.csv
  - Provides a reference dataset containing start and end dates along with the modified files. It is used to compare expected results with actual outputs. You can test various scenarios by customizing this file.
  ### config.json
  - Allows you to specify the repository to be tested, the local path where it should be cloned, the branch to analyze, and the start and end dates for conducting tests within a specified time frame.
  ### Docker 
  - Sets up the necessary environment with Python, Git, dependencies, and your project files to run the defect predictor seamlessly inside a container.