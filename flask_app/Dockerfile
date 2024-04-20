# Here we use a Python 3.9 image
FROM python:3.9-slim

# We set the working directory to /app
WORKDIR /app

# Here we install all dependencies for the application
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# Install the large packages individually to better use Docker's cache
RUN pip install --no-cache-dir --default-timeout=500 torch==2.2.0
RUN pip install --no-cache-dir -r requirements.txt

# 8080 is the required port for GCP
ENV PORT=8080

# Run gunicorn when the container launches
# Here we retrieve the PORT environment variable in the command itself
CMD ["sh", "-c", "gunicorn -b :$PORT run:app"]
