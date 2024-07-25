FROM python:3.11.4-slim

# Set the working directory
WORKDIR /app

# Install git and any other required system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy the requirements.txt file first to install dependencies
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir && \
    rm -rf /root/.cache/pip

COPY . .