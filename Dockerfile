# Use a slim version to reduce image size and potential network issues
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy application files to the container
COPY . .

# Install system dependencies for TensorFlow
# RUN apt-get update && \
#     apt-get install -y libopenblas-dev liblapack-dev && \
#     rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies with a longer timeout
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install tensorflow==2.12.0

# Expose port 5000 for Flask
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
