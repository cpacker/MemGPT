# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at the working directory
COPY . .

ENV OPENAI_API_KEY=""

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "main.py"]
