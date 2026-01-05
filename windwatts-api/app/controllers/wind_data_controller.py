from typing import List, Optional
from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import StreamingResponse
import zipfile
import tempfile
import re
import os

from app.config_manager import ConfigManager
from app.config.model_config import MODEL_CONFIG
from app.data_fetchers.s3_data_fetcher import S3DataFetcher
from app.data_fetchers.athena_data_fetcher import AthenaDataFetcher
from app.data_fetchers.data_fetcher_router import DataFetcherRouter
from app.utils.data_fetcher_utils import format_coordinate, chunker
from app.utils.validation import validate_model, validate_limit
from app.utils.wind_data_core import (
    get_windspeed_core,
    get_production_core,
    get_timeseries_core,
)

from app.power_curve.global_power_curve_manager import power_curve_manager
from app.schemas import (
    WindSpeedResponse,
    AvailablePowerCurvesResponse,
    EnergyProductionResponse,
    NearestLocationsResponse,
    TimeseriesBatchRequest,
    ModelInfoResponse,
)

router = APIRouter()

# Initialize DataFetcherRouter
data_fetcher_router = DataFetcherRouter()
_skip_data_init = os.environ.get("SKIP_DATA_INIT", "0") == "1"

# Initialize data fetchers
athena_data_fetchers = {}
s3_data_fetchers = {}

if not _skip_data_init:
    # Initialize ConfigManager
    config_manager = ConfigManager(
        secret_arn_env_var="WINDWATTS_DATA_CONFIG_SECRET_ARN",
        local_config_path="./app/config/windwatts_data_config.json",
    )
    athena_config = config_manager.get_config()

    # Initialize Athena data fetchers
    athena_data_fetchers["era5"] = AthenaDataFetcher(
        athena_config=athena_config, source_key="era5"
    )
    athena_data_fetchers["ensemble"] = AthenaDataFetcher(
        athena_config=athena_config, source_key="ensemble"
    )
    athena_data_fetchers["wtk"] = AthenaDataFetcher(
        athena_config=athena_config, source_key="wtk"
    )

    # Initialize S3 data fetchers
    s3_data_fetchers["era5"] = S3DataFetcher(
        bucket_name="windwatts-era5",
        prefix="era5_timeseries",
        grid="era5",
        s3_key_template="era5",
    )
    s3_data_fetchers["wtk"] = S3DataFetcher(
        bucket_name="wtk-led", prefix="1224", grid="wtk", s3_key_template="wtk"
    )

    # Register fetchers with DataFetcherRouter
    # Register with simple names: athena, s3 (not athena_era5, s3_era5)
    for model_key in ["era5", "ensemble", "wtk"]:
        if model_key in athena_data_fetchers:
            data_fetcher_router.register_fetcher(
                f"athena_{model_key}", athena_data_fetchers[model_key]
            )
        if model_key in s3_data_fetchers:
            data_fetcher_router.register_fetcher(
                f"s3_{model_key}", s3_data_fetchers[model_key]
            )


# API Endpoints
@router.get(
    "/{model}/windspeed",
    summary="Retrieve wind speed data for a location",
    response_model=WindSpeedResponse,
    responses={
        200: {
            "description": "Wind speed data retrieved successfully",
            "model": WindSpeedResponse,
        },
        400: {"description": "Bad request - invalid parameters"},
        404: {"description": "Data not found"},
        500: {"description": "Internal server error"},
    },
)
def get_windspeed(
    model: str = Path(..., description="Data model: era5, wtk, or ensemble"),
    lat: float = Query(..., description="Latitude of the location"),
    lng: float = Query(..., description="Longitude of the location"),
    height: int = Query(..., description="Height in meters"),
    period: str = Query(
        "all", description="Time period: all, annual, monthly, hourly (varies by model)"
    ),
    source: Optional[str] = Query(
        None,
        description="Data source: athena or s3. Defaults to model's default source (athena).",
    ),
):
    """
    Retrieve wind speed data for a specific location and height.

    - **model**: Data model (era5, wtk, ensemble)
    - **lat**: Latitude (varies by model, refer info endpoint for coordinate bounds)
    - **lng**: Longitude (varies by model, refer info endpoint for coordinate bounds)
    - **height**: Height in meters (varies by model, refer info endpoint for the available heights)
    - **period**: Time aggregation period (default: all)
    - **source**: Optional data source override
    """
    try:
        # Catch invalid model before core function call
        model = validate_model(model)

        # Use default source if not provided
        if source is None:
            source = MODEL_CONFIG.get(model, {}).get("default_source", "athena")

        return get_windspeed_core(
            model, lat, lng, height, period, source, data_fetcher_router
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/{model}/production",
    summary="Get energy production estimate for a location with a power curve",
    response_model=EnergyProductionResponse,
    responses={
        200: {
            "description": "Energy production data retrieved successfully",
            "model": EnergyProductionResponse,
        },
        400: {"description": "Bad request - invalid parameters"},
        404: {"description": "Data not found"},
        500: {"description": "Internal server error"},
    },
)
def get_production(
    model: str = Path(..., description="Data model: era5, wtk, or ensemble"),
    lat: float = Query(..., description="Latitude of the location"),
    lng: float = Query(..., description="Longitude of the location"),
    height: int = Query(..., description="Height in meters"),
    powercurve: str = Query(
        ..., description="Power curve identifier (e.g., nrel-reference-100kW)"
    ),
    period: str = Query(
        "all",
        description="Time period: all, summary, annual, monthly (varies by model)",
    ),
    source: Optional[str] = Query(
        None,
        description="Data source: athena or s3. Defaults to model's default source (athena).",
    ),
):
    """
    Retrieve energy production estimates for a specific location, height, and power curve.

    - **model**: Data model (era5, wtk, ensemble)
    - **lat**: Latitude (varies by model, refer info endpoint for coordinate bounds)
    - **lng**: Longitude (varies by model, refer info endpoint for coordinate bounds)
    - **height**: Height in meters (varies by model, refer info endpoint for the available heights)
    - **powercurve**: Power curve to use for calculations
    - **period**: Time aggregation period (default: all)
    - **source**: Optional data source override
    """
    try:
        # Catch invalid model before core function call
        model = validate_model(model)

        # Use default source if not provided
        if source is None:
            source = MODEL_CONFIG.get(model, {}).get("default_source", "athena")

        return get_production_core(
            model, lat, lng, height, powercurve, period, source, data_fetcher_router
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/powercurves",
    summary="Fetch all available power curves",
    response_model=AvailablePowerCurvesResponse,
    responses={
        200: {
            "description": "Available power curves retrieved successfully",
            "model": AvailablePowerCurvesResponse,
        },
        500: {"description": "Internal server error"},
    },
)
def get_powercurves():
    """
    Retrieve a list of all available power curves.

    Power curves are model-agnostic and can be used with any dataset (era5, wtk, ensemble).
    """
    try:
        all_curves = list(power_curve_manager.power_curves.keys())

        def extract_kw(curve_name: str):
            # Extracts the kw value from nrel curves, e.g. "nrel-reference-2.5kW" -> 2.5
            match = re.search(r"nrel-reference-([0-9.]+)kW", curve_name)
            if match:
                return float(match.group(1))
            return float("inf")

        nrel_curves = [c for c in all_curves if c.startswith("nrel-reference-")]
        other_curves = [c for c in all_curves if not c.startswith("nrel-reference-")]

        nrel_curves_sorted = sorted(nrel_curves, key=extract_kw)
        other_curves_sorted = sorted(other_curves)

        ordered_curves = nrel_curves_sorted + other_curves_sorted
        return {"available_power_curves": ordered_curves}
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/{model}/grid-points",
    summary="Find nearest grid locations to a coordinate",
    response_model=NearestLocationsResponse,
    responses={
        200: {"description": "Nearest grid locations retrieved successfully"},
        400: {"description": "Bad request - invalid parameters"},
        500: {"description": "Internal server error"},
    },
)
def get_grid_points(
    model: str = Path(..., description="Data model: era5, wtk, or ensemble"),
    lat: float = Query(..., description="Latitude of the target location"),
    lng: float = Query(..., description="Longitude of the target location"),
    limit: int = Query(1, description="Number of nearest grid points to return (1-4)"),
    source: Optional[str] = Query(
        None, description="Data source. Defaults to model's default source."
    ),
):
    """
    Find the nearest grid points to a given coordinate.

    Returns grid indices and their coordinates for the closest data points in the model's grid.

    - **model**: Data model (era5, wtk, ensemble)
    - **lat**: (varies by model, refer info endpoint for coordinate bounds)
    - **lng**: (varies by model, refer info endpoint for coordinate bounds)
    - **limit**: Number of nearest points to return (1-4)
    - **source**: Optional data source override
    """
    try:
        model = validate_model(model)

        # Grid lookup only available via athena
        # Use athena fetcher for the specified model
        fetcher = athena_data_fetchers.get(model)

        if not fetcher or not hasattr(fetcher, "find_nearest_locations"):
            raise HTTPException(
                status_code=400,
                detail=f"Grid point lookup not available for model '{model}'",
            )

        # Call find_nearest_locations on the fetcher
        limit = validate_limit(limit)
        result = fetcher.find_nearest_locations(lat=lat, lng=lng, n_neighbors=limit)

        locations = [
            {"index": str(i), "latitude": float(a), "longitude": float(o)}
            for i, a, o in result
        ]

        return {"locations": locations}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/{model}/info",
    summary="Get metadata information about a data model",
    response_model=ModelInfoResponse,
    responses={
        200: {"description": "Model information retrieved successfully"},
        400: {"description": "Invalid model"},
        500: {"description": "Internal server error"},
    },
)
def get_model_info(
    model: str = Path(..., description="Data model: era5, wtk, or ensemble"),
):
    """
    Retrieve metadata and configuration information about a specific data model.

    Returns information about:
    - Supported time periods
    - Available years for timeseries downloads
    - Available heights
    - Grid information (model’s geographic coverage and spatial-temporal resolution)
    - Links & References

    - **model**: Data model (era5, wtk, ensemble)
    """
    try:
        model = validate_model(model)
        config = MODEL_CONFIG[model]

        return {
            "model": model,
            # "available_sources": config["sources"],
            # "default_source": config["default_source"],
            "supported_periods": config["period_type"],
            "available_years": config.get("years", {}).get("full", []),
            "available_heights": config.get("heights", []),
            "grid_info": config.get("grid_info", {}),
            "links": config.get("links", []),
            "references": config.get("references", []),
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/{model}/timeseries",
    summary="Download timeseries CSV data for a specific grid point",
    responses={
        200: {"description": "CSV file downloaded successfully"},
        400: {"description": "Bad request - invalid parameters"},
        404: {"description": "Data not found"},
        500: {"description": "Internal server error"},
    },
)
def download_timeseries(
    model: str = Path(..., description="Data model: era5 or wtk"),
    gridIndex: str = Query(..., description="Grid index identifier"),
    years: Optional[List[int]] = Query(
        None, description="Years to download (defaults to sample years)"
    ),
    source: str = Query(
        "s3",
        description="Data source: athena or s3 (typically s3 for timeseries downloads)",
    ),
):
    """
    Download timeseries data as CSV for a specific grid point.

    Returns raw timeseries data for the specified grid index and years.

    - **model**: Data model (era5, wtk)
    - **gridIndex**: Grid index from grid-points endpoint
    - **years**: List of years to include (optional)
    - **source**: Data source (defaults to S3)
    """
    try:
        # Get CSV content from core function
        csv_content = get_timeseries_core(
            model, [gridIndex], years, source, data_fetcher_router
        )

        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="wind_data_{gridIndex}.csv"'
            },
        )

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/{model}/timeseries/batch",
    summary="Download multiple timeseries CSVs as a ZIP file",
    responses={
        200: {"description": "ZIP file downloaded successfully"},
        400: {"description": "Bad request - invalid parameters"},
        404: {"description": "Data not found"},
        500: {"description": "Internal server error"},
    },
)
def download_timeseries_batch(
    payload: TimeseriesBatchRequest,
    model: str = Path(..., description="Data model: era5 or wtk"),
):
    """
    Download timeseries data for multiple grid points as a ZIP archive.

    Accepts a request body with grid locations, optional years, and data source.
    Returns a ZIP file containing CSV files for each location.

    - **model**: Data model (era5, wtk)
    - **payload**: Request body containing:
      - **locations**: List of grid locations with indices (use grid-points endpoint)
      - **years**: List of years to include (optional, defaults to sample years)
      - **source**: Data source (optional, defaults to s3)
    """
    try:
        # Create spooled temporary file for ZIP
        spooled = tempfile.SpooledTemporaryFile(max_size=30 * 1024 * 1024, mode="w+b")

        with zipfile.ZipFile(spooled, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for loc in payload.locations:
                csv_content = get_timeseries_core(
                    model,
                    [loc.index],
                    payload.years,
                    payload.source,
                    data_fetcher_router,
                )
                file_name = f"wind_data_{format_coordinate(loc.latitude)}_{format_coordinate(loc.longitude)}.csv"
                zf.writestr(file_name, csv_content)

        spooled.seek(0)

        headers = {
            "Content-Disposition": f'attachment; filename="wind_data_{model}_{len(payload.locations)}_points.zip"'
        }

        return StreamingResponse(
            chunker(spooled), media_type="application/zip", headers=headers
        )

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
