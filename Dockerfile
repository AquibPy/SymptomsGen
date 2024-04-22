# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the project files to the container
COPY . .

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the environment variable for the configuration file
ENV CONFIG_FILE=/app/config.py

# Expose the port for the FastAPI application
EXPOSE 8080

# Start the FastAPI application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]