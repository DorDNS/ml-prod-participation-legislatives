# Slim base image: smaller download and fewer packages to patch
FROM python:3.12-slim

# Send logs straight to stdout so they show up immediately in the platform logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first so Docker caches the install layer
# when only the application code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only the application code goes into the image
COPY app/ ./app/

# Run as a non-root user to limit the impact of a compromise
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# The port comes from the PORT variable (set by Render), 8000 by default.
# "exec" lets uvicorn receive the stop signal from Docker.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]