# Use Python 3.11 base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the necessary Python files into the container
COPY incremental_predictor.py /app
COPY requirements.txt /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for the web interface if needed (optional)
EXPOSE 8501

# Command to run your Python script
CMD ["python3", "incremental_predictor.py"]
