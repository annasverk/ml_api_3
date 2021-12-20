# Build an image starting with the Python 3.8 image
FROM python:3.8

# Set working directory
WORKDIR /mlflow/

# Install mlflow
RUN pip install mlflow==1.22.0
RUN chmod 777 -R /mlflow/

# Add metadata to the image to describe that the container is listening on port 5050
EXPOSE 5050

# Set environment variables
ENV BACKEND_URI sqlite:////mlflow/mlflow.db
ENV ARTIFACT_ROOT /mlflow/artifacts

# Run mlflow server
CMD mlflow server --host 0.0.0.0 --port 5050 --file-store /mlflow/