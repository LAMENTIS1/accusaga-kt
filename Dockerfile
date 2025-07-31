# Use an official Python runtime as the base ima# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    gettext \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download the wait-for-it.sh script (if it's not already in your codebase)
RUN curl -o /app/wait-for-it.sh https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh \
    && chmod +x /app/wait-for-it.sh

# Copy the current directory contents into the container at /app
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8008

# Default command to run the Django app (will be overridden by docker-compose command)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8008"]
