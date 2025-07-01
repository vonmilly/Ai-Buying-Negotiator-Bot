# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for llama.cpp and other tools
# build-essential for 'make', cmake for llama.cpp, git for cloning
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Copy the llama.cpp source and build it
# Ensure you have placed your mistral.gguf model in llama.cpp/models/ on your host
# before building this Docker image.
COPY llama.cpp /app/llama.cpp
WORKDIR /app/llama.cpp
RUN make

# Return to the app directory
WORKDIR /app

# Copy the FastAPI application files
COPY app/main.py .
COPY app/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

