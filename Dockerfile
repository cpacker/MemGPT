# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Set default environment variables
ENV OPENAI_API_KEY=""

# This ensures that the pip install layer is recreated only when there are changes in the requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# After the dependencies have been installed, copy the rest of the application's files into the container
COPY . .

# Start CLI
ENTRYPOINT ["python", "main.py"]
