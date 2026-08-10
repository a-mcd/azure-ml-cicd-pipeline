# Dockerfile
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest
RUN apt-get update && apt-get install -y libgl1-mesa-glx
COPY requirements.txt .
RUN pip install -r requirements.txt
