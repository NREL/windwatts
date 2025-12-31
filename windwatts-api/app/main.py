from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.schemas import HealthCheckResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from app.controllers.wtk_data_controller import router as wtk_data_router
from app.controllers.era5_data_controller import router as era5_data_router
from app.controllers.wind_data_controller import router as wind_data_router
from app.middleware import AuditMiddleware, LoggingMiddleware
from app.exception_handlers import log_unhandled_exceptions, log_validation_errors
from textwrap import dedent

app = FastAPI(
    title="WindWatts API",
    version="1.0.0",
    root_path="/api",
    description=dedent(
        """
        Welcome to NLR's WindWatts API.

        - Rate limits: tiered - 10 / 100 / 1000 requests per minute per IP.
        - Base path: `/api`
        - Contact: windwatts@nrel.gov

        ## API Versions

        **v1 (Recommended)**:
        - `/v1/{model}/windspeed` - Wind speed data
        - `/v1/{model}/production` - Energy production estimates
        - `/v1/{model}/timeseries` - Raw timeseries downloads
        - Supports models: `era5`, `wtk`, `ensemble`

        **Legacy**: Model-specific endpoints (deprecated)
        - `/wtk/*` - WTK-specific endpoints
        - `/era5/*` - ERA5-specific endpoints

        Use the endpoints below to retrieve wind resource and production estimates.
        """
    ).strip(),
)

app.add_middleware(LoggingMiddleware)  # Logging middleware first
app.add_middleware(AuditMiddleware)  # Audit middleware second

app.add_exception_handler(Exception, log_unhandled_exceptions)
app.add_exception_handler(RequestValidationError, log_validation_errors)

origins = [
    "http://localhost",
    "https://windwatts2-dev.stratus.nrel.gov",
    "https://windwatts2-stage.stratus.nrel.gov",
    "https://windwatts2-prod.stratus.nrel.gov",
    "https://windwatts.nrel.gov",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API v1
app.include_router(wind_data_router, prefix="/v1", tags=["v1-wind-data"])

# Legacy routes - Deprecated
# TODO: Remove these routes
app.include_router(
    wtk_data_router, prefix="/wtk", tags=["wtk-data (deprecated)"], deprecated=True
)
app.include_router(
    era5_data_router, prefix="/era5", tags=["era5-data (deprecated)"], deprecated=True
)


@app.get("/healthcheck", response_model=HealthCheckResponse)
def healthcheck():
    return JSONResponse({"status": "up"}, status_code=200)


# Serve generated OpenAPI JSON if present
@app.get("/openapi.json", include_in_schema=False, response_model=None)
def serve_openapi_json():
    return app.openapi()


handler = Mangum(app)
