FROM nvcr.io/nvidia/pytorch:26.03-py3

# Set the working directory inside the container
WORKDIR /app

# Upgrade pip to ensure smooth package installations
RUN pip install --no-cache-dir --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt .

# Install all project dependencies (vLLM, Transformers, Accelerate, etc.)
RUN pip install --no-cache-dir -r requirements.txt