# System Overview

## Architecture

```mermaid
c4context
      title System Context Diagram for WindWatts

      person(user, "User", "A user wanting to access wind data")
      system_boundary(windwatts, "WindWatts System") {
        container(ui, "Frontend UI", "React, Vite", "Provides the web interface for users")
        container(api, "Backend API", "FastAPI, Python", "Handles API requests and business logic")
        containerDb(db, "Database", "PostgreSQL", "Stores application data")
        container(windwatts_data, "Core Library", "Python", "Scientific data processing logic")
      }
      system_ext(aws, "AWS Services", "S3, Athena, Glue", "Stores and queries large-scale wind data")

      rel(user, ui, "Uses", "HTTPS")
      rel(ui, api, "Makes API calls to", "JSON/HTTPS")
      rel(api, db, "Reads/Writes", "SQL/TCP")
      rel(api, windwatts_data, "Imports", "Python Module")
      rel(api, aws, "Queries", "Boto3/HTTPS")
```

WindWatts is a monorepo consisting of:

- **Frontend**: React application (`windwatts-ui`)
- **Backend**: Python FastAPI application (`windwatts-api`)
- **Core Library**: Data processing logic (`windwatts_data`)

## Tech Stack

### Frontend

- **Language**: TypeScript / JavaScript
- **Framework**: React 19
- **Build Tool**: Vite
- **UI Library**: Material UI v7
- **Runtime**: Node.js >= 22.14.0

### Backend

- **Language**: Python 3.13
- **Framework**: FastAPI [standard]
- **Server**: Gunicorn / Uvicorn
- **ORM**: SQLAlchemy
- **Database**: PostgreSQL

### Infrastructure

- **Containerization**: Docker
- **Cloud Provider**: AWS (S3, Athena, Glue)
