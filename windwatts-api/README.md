# WindWatts API (Backend)

This directory contains the Python backend for WindWatts, built with FastAPI.

For development instructions, setup guides, and architecture details, please see the **[Backend Documentation](../docs/03-backend.md)**.

This project is set up like a standard Python project. The initialization
process also creates a virtualenv within this project, stored under the `.venv`
directory. To create the virtualenv it assumes that there is a `python3`
(or `python` for Windows) executable in your path with access to the `venv`
package. If for any reason the automatic creation of the virtualenv fails,
you can create the virtualenv manually.

The project has a Makefile with the following targets:

- `make setup` initializes the project
- `make run` runs the webservice layer locally for testing

To manually create a virtualenv on MacOS and Linux:

```
$ python3 -m venv .venv
```

After the init process completes and the virtualenv is created, you can use the following
step to activate your virtualenv.

```
$ source .venv/bin/activate
```

If you are a Windows platform, you would activate the virtualenv like this:

```
% .venv\Scripts\activate.bat
```

Once the virtualenv is activated, you can install the required dependencies.

```
$ pip install -r requirements.txt
```

## Useful commands

- `uvicorn app.main:app --reload` runs the webservice layer locally for testing

## Database Configuration

### Local Development
The project uses PostgreSQL for data storage. For local development:

1. Add the following to `.env` in the api root:
```.env
DATABASE_URL=postgresql://windwatts:windwatts@postgres:5432/windwatts_db
POSTGRES_USER=windwatts
POSTGRES_PASSWORD=windwatts
POSTGRES_DB=windwatts_db
```

2. Start the services using docker-compose:
```bash
docker-compose up
```

The local PostgreSQL instance will be available at:
- Host: postgres
- Port: 5432
- Database: windwatts_db
- Username: windwatts
- Password: windwatts

### Production Deployment TODOs
For production deployment, the following items need to be configured by the cloud team:

1. Set up an RDS PostgreSQL instance in AWS
2. Configure the following in AWS Secrets Manager:
   - Database connection string (DATABASE_URL)
   - Other necessary database credentials
3. Update the appfleet configuration to:
   - Remove the local postgres service
   - Include the AWS Secrets Manager ARN for database configuration
   - Configure necessary IAM roles for RDS access

Enjoy!
