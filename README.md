# Defect Predictor using RepoMiner

This project provides an implementation of a Defect Predictor by leveraging RepoMiner and PyDriller to analyze commit history and predict the risk associated with files in a given revision. It extends the RepoMiner example and includes risk calculation based on commit history.

# Setup Instructions


## Installation
Follow the steps below to set up and run the project after cloning its repo

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

  
### 4. Login and run 
Create a docker container and run a bash shell in it. From there, you can modify the file config.json as you wish.
```bash
 docker run -it --rm defect-predictor bash
```

## Overview of files

  ### SampleDefectPredictor.py:
  - This will identify all Python files that are going to be analyzed for bugginess within a date. From there, we also calculate a risk score (this is incomplete at the moment).
  ### harness.py
  - This script runs the defect predictor (SampleDefectPredictor.py) logic to generate a list of modified files.
  ### requirements.txt
  - Lists all the dependencies and libraries required to run the project seamlessly.
  ### ground_truth.csv
  - Provides a reference dataset containing start and end dates along with the modified files. It is used to compare expected results with actual outputs.
  ### config.json
  - Allows you to specify the repository to be tested, the local path where it should be cloned, the branch to analyze, and the start and end dates.
  ### Docker 
  - Sets up the necessary environment with Python, Git, dependencies, and your project files to run the defect predictor seamlessly inside a container.
