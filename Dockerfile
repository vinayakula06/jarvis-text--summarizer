# Use an official Python runtime as a parent image
FROM python:3.9-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install PyTorch first with extended timeout (most likely to timeout)
RUN pip install --no-cache-dir --timeout=1000 torch==1.10.2

# Install remaining packages
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Set NLTK data path and download punkt
ENV NLTK_DATA=/app/nltk_data
RUN mkdir -p ${NLTK_DATA} && python -c "import nltk; nltk.download('punkt', download_dir='${NLTK_DATA}')"

# Expose the port the app runs on
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP=app.py

# Run the application
CMD ["flask", "run", "--host=0.0.0.0"]
