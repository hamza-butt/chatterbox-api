# Use a Python 3.10 base image (Chatterbox requires >= 3.10)
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files
# and to ensure stdout/stderr are unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for building some Python packages and audio processing
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Upgrade pip and install setuptools (pinning setuptools < 70 for the pkg_resources issue)
RUN pip install --upgrade pip "setuptools<70.0.0" wheel

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the API port
EXPOSE 5001

# Command to run the application
CMD ["python", "app.py"]
