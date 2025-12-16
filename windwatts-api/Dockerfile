# Use the official Python image with Debian bookworm
FROM python:3.13-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
ENV PROJ_LIB=/usr/share/proj
ENV PYTHONPATH=/app

# Set the working directory
WORKDIR $APP_HOME

# Install system dependencies and upgrade packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    gdal-bin libgdal-dev libproj-dev libgeos-dev \
    gcc python3-dev libffi-dev curl

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download and install windwatts_data.whl from s3
ARG WINDWATTS_DATA_URL
ARG WINDWATTS_DATA_VERSION=1.0.4.1
ARG WINDWATTS_DATA_FILE=windwatts_data-${WINDWATTS_DATA_VERSION}-py3-none-any.whl
ENV WINDWATTS_DATA_URL=${WINDWATTS_DATA_URL}
RUN curl -o /tmp/windwatts_data-${WINDWATTS_DATA_VERSION}-py3-none-any.whl ${WINDWATTS_DATA_URL}${WINDWATTS_DATA_FILE} && \
    pip install --no-cache-dir /tmp/${WINDWATTS_DATA_FILE} && \
    rm /tmp/${WINDWATTS_DATA_FILE}

# Copy the FastAPI application
COPY . .

# Generate OpenAPI spec at build-time
RUN python scripts/generate_openapi.py || true

# Expose the port
EXPOSE 8000

# Run the app with Gunicorn
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "app.main:app"]
