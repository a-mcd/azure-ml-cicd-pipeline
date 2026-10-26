# Azure ML training image, registered as the training environment from
# azureml/docker.environment.template.yaml.
#
# The base image tag is pinned so a rebuild cannot pick up a different image.
# Base image Python: 3.12.15, the same minor version as CI. Checked with:
#   docker run --rm <image> python --version
# Re-check and update this line when changing the tag;
# tests/unit/test_dependency_pins.py compares it with CI's Python version.
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:20261018.v1
RUN apt-get update && apt-get install -y libgl1-mesa-glx
COPY requirements.txt .
RUN pip install -r requirements.txt
