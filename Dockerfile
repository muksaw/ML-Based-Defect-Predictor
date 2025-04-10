FROM python:3.9-slim

# Dependencies:
#   Python version **<3.9**.
#   Git version **>=2.38.0**.

# Set the working directory inside the container
WORKDIR /app

# Install git and build essentials, and clean up in the same layer
RUN apt-get update && \
    apt-get install -y \
    git \
    build-essential \
    gcc \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies with pip no-cache and combine SpaCy download
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Create the Testing directory
RUN mkdir -p /app/Testing

# Copy application files
COPY . .

# Default command
CMD ["python", "ml_harness.py", "--train", "--predict"]

# Note: Output files will be saved to the ./outputs directory in the repository
# When running the container, mount the outputs directory to persist the results:
# docker run --rm -it -v $(pwd)/outputs:/app/outputs defect-predictor
