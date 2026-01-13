FROM public.ecr.aws/lambda/python:3.9

# Copy requirements and install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py ${LAMBDA_TASK_ROOT}/

# Set the handler
CMD ["app.handler"]
