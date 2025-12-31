# Quickstart Guide

This guide will help you get the WindWatts application up and running on your local machine.

## Prerequisites

1.  **Docker**: Ensure Docker is installed and running.
2.  **AWS Credentials**: You need access to the WindWatts Data package.
3.  **Google Maps API Key**: Required for the frontend map.

## Option 1: Docker (Recommended)

This sets up both the API and the UI with a local database.

### 1. Clone the Repository

```shell
git clone https://github.com/NREL/windwatts.git
cd windwatts/
git checkout develop # Ensure you are on the 'develop' branch (the main development branch)
```

### 2. Configure Environment

**Root directory (`.env`)**:
Create a `.env` file in the root:

```plaintext
WINDWATTS_DATA_URL=https://windwatts-era5.s3.us-west-2.amazonaws.com/
AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
AWS_SESSION_TOKEN="YOUR_AWS_SESSION_TOKEN"
```

**Frontend (`windwatts-ui/.env.development`)**:

```plaintext
VITE_API_BASE_URL=http://windwatts-proxy:80
VITE_MAP_API_KEY=YOUR_MAP_API_KEY
VITE_MAP_ID=YOUR_MAP_ID
```

**Backend (`windwatts-api/app/config/windwatts_data_config.json`)**:

```json
{
  "region_name": "us-west-2",
  "output_location": "S3_BUCKET_URI_FOR_ATHENA_RESULTS",
  "output_bucket": "NAME_OF_S3_BUCKET_FOR_ATHENA_RESULTS",
  "database": "NAME_OF_THE_GLUE_DATABASE",
  "athena_workgroup": "NAME_OF_THE_ATHENA_WORKGROUP",
  "sources": {
    "wtk": {
      "bucket_name": "NAME_OF_THE_WTK_S3_BUCKET",
      "athena_table_name": "NAME_OF_THE_ATHENA_TABLE_FOR_WTK",
      "alt_athena_table_name": ""
    },
    "era5": {
      "bucket_name": "NAME_OF_THE_ERA5_S3_BUCKET",
      "athena_table_name": "NAME_OF_THE_ATHENA_TABLE_FOR_ERA5",
      "alt_athena_table_name": ""
    }
  }
}
```

### 3. Run the App

```shell
docker compose up --build
```

To clean up:

```shell
docker compose down --volumes --remove-orphans
```

### 4. Access

- **UI**: http://localhost:5173/
- **API**: http://localhost:8080

## Option 2: Native Setup (Without Docker)

If you prefer running services directly:

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

See [Backend Documentation](03-backend.md) and [Frontend Documentation](04-frontend.md) for more details on native development.
