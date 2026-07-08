# AWS Lambda Python 3.12 base image
FROM public.ecr.aws/lambda/python:3.12

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your application code and models
COPY app/ ./app/
COPY models/ ./models/

# Set the CMD to your handler
CMD ["app.main.handler"]
