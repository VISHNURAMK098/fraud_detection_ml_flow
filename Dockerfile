# Base image
FROM public.ecr.aws/lambda/python:3.10

# Set working directory inside container
WORKDIR /var/task

# Install system packages and Python dependencies
RUN yum install -y \
    git \
    curl \
    && yum clean all

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all source files
COPY . .

# Expose MLflow UI port (optional, if you want to run it from inside the container)
EXPOSE 5000

# Run the main pipeline
CMD ["main.lambda_handler"]
