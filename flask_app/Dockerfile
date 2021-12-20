# Build an image starting with the Python 3.8 image
FROM python:3.8

# Set environment variables
ENV HOST 0.0.0.0
ENV PORT 8080
ENV DEBUG False
ENV MLFLOW_HOST mlflow
ENV MLFLOW_PORT 5050

ENV LC_ALL en_US.utf-8
ENV LANG en_US.utf-8

# Copy all files to the directory and set it as the working directory
COPY . /flask_app
WORKDIR /flask_app

# Install dependencies
RUN pip install -r requirements.txt

# Add metadata to the image to describe that the container is listening on port 8080
EXPOSE 8080

# Run
ENTRYPOINT ["python"]
CMD ["app.py"]