# Start with the official Airflow image
FROM apache/airflow:2.8.1

# Copy our requirements file into the image
COPY requirements.txt /

# Install the Python packages
RUN pip install --no-cache-dir -r /requirements.txt