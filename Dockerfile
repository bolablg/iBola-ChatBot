
# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install cron
RUN apt-get update && apt-get -y install cron

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# Add the cron job
RUN echo "0 0 * * * root python /app/pipeline/sync.py >> /var/log/cron.log 2>&1" > /etc/cron.d/sync-cron

# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/sync-cron

# Create the log file to be able to run tail
RUN touch /var/log/cron.log

# Expose the port the app runs on
EXPOSE 8000

# Start the cron daemon in the background and run the app
CMD cron && tail -f /var/log/cron.log & uvicorn app.main:app --host 0.0.0.0 --port 8000
