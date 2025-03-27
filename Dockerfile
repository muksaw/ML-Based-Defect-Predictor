FROM python:3.9-slim

# Dependencies:
#   Python version **<3.9**.
#   Git version **>=2.38.0**.

# Set the working directory inside the container
WORKDIR /app

# Install git and basic utilities
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY *.py ./
COPY config.json ./

# Download the SpaCy language model
RUN python -m spacy download en_core_web_sm

# Create the Testing directory as a concrete path
RUN mkdir -p /app/Testing

# Create a directory for output files
RUN mkdir -p /app/outputs

# Copy the entire project directory into the container
COPY . .

# Uncomment the command below to execute your script automatically
CMD ["python", "ml_harness.py", "--train", "--predict", "--output-file", "/app/outputs/output.txt"]

# To access output files, run the container with a volume mount:
# docker run --rm -it -v $(pwd)/outputs:/app/outputs defect-predictor
