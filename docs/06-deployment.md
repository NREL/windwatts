# Deployment Guide

## Deploy as a Container

This assumes Docker is available on the host.

### Build

```shell
docker build -t tap-api:latest .
```

### Run (Production Mode)

```shell
docker run -p 8080:80 -it tap-api:latest uvicorn app.main:app --host 0.0.0.0 --port 80 --workers 4
```

Inside the container, the FastAPI app runs on port `80`. On the host, it's mapped to `8080`.

### Troubleshooting

To explore the container:

```shell
docker run -p 8080:80 -it tap-api:latest /bin/bash
```

## Production Configuration

### Database

For production deployment (e.g., AWS):

1.  **RDS**: Set up an RDS PostgreSQL instance.
2.  **Secrets Manager**: Configure `DATABASE_URL` and credentials in AWS Secrets Manager.
3.  **App Configuration**: Update the service configuration to use the RDS instance instead of a local container.

### Environment Variables

Ensure the production environment has the correct `.env` values (similar to the root `.env` but with production endpoints).

