# Backend Development (API)

The backend is a FastAPI application connected to a PostgreSQL database and AWS Athena/S3 for data retrieval.

## Setup

The project uses a standard Python structure.

### Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### WindWatts Data Dependency

The `windwatts_data` package is a large dependency stored in S3. It is not in PyPI.
If you are running locally without Docker, you must download and install it manually:

```bash
# Download the wheel from S3 (check Dockerfile for exact URL/version)
curl -O https://windwatts-era5.s3.us-west-2.amazonaws.com/windwatts_data-1.0.4-py3-none-any.whl

# Verify the SHA256 checksum for integrity
# Expected checksum should be obtained from a trusted source (e.g., project maintainers)
# Example verification:
sha256sum windwatts_data-1.0.4-py3-none-any.whl
# Compare the output with the expected checksum before proceeding

# Install the wheel after verification
pip install windwatts_data-1.0.4-py3-none-any.whl
```

**Security Note**: Always verify the integrity of the downloaded wheel file before installation. Contact the project maintainers to obtain the expected SHA256 checksum for the version you're installing. This protects against compromised or modified files.

### Makefile Targets

-   `make setup`: Initializes the project.
-   `make run`: Runs the webservice locally.

## Development

### Running Locally

```bash
uvicorn app.main:app --reload
```

Or using the wrapper:
```bash
python api.py -d
```

### Database

**Local Development**:
Add to `.env` in `windwatts-api/`:

```plaintext
DATABASE_URL=postgresql://windwatts:windwatts@postgres:5432/windwatts_db
POSTGRES_USER=windwatts
POSTGRES_PASSWORD=windwatts
POSTGRES_DB=windwatts_db
```

Use Docker Compose (from root) to start the database.

## Testing

To run tests:
```bash
pytest
```

## API Documentation

When the app is running, visit `/docs` (e.g., `http://localhost:8080/docs`) to see the auto-generated Swagger UI.
