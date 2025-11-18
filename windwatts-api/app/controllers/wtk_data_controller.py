from typing import Optional
from fastapi import APIRouter, HTTPException, Path, Query
import re
import os
from typing import List
import io
from fastapi.responses import StreamingResponse
import zipfile
import tempfile
# commented out the data functions until I can get local athena_config working
from app.config_manager import ConfigManager
from app.data_fetchers.s3_data_fetcher import S3DataFetcher
from app.data_fetchers.athena_data_fetcher import AthenaDataFetcher
# from app.data_fetchers.database_data_fetcher import DatabaseDataFetcher
from app.data_fetchers.data_fetcher_router import DataFetcherRouter
# from app.database_manager import DatabaseManager
from app.utils.data_fetcher_utils import format_coordinate, chunker

from app.power_curve.global_power_curve_manager import power_curve_manager
from app.schemas import (
    WindSpeedResponse,
    AvailablePowerCurvesResponse,
    EnergyProductionResponse,
    GridLocation,
    NearestLocationsResponse
)

router = APIRouter()

# Create router first, then optionally initialize heavy dependencies unless skipped
data_fetcher_router = DataFetcherRouter()
_skip_data_init = os.environ.get("SKIP_DATA_INIT", "0") == "1"
if not _skip_data_init:
    # Initialize ConfigManager
    config_manager = ConfigManager(
        secret_arn_env_var="WINDWATTS_DATA_CONFIG_SECRET_ARN",
        local_config_path="./app/config/windwatts_data_config.json")  # replace with YOUR local config path
    athena_config = config_manager.get_config()

# Initialize DataFetchers
s3_data_fetcher_wtk = S3DataFetcher(bucket_name="wtk-led", prefix="1224", grid="wtk", s3_key_template="wtk")
athena_data_fetcher_wtk = AthenaDataFetcher(athena_config=athena_config, source_key='wtk')
# db_manager = DatabaseManager()
# db_data_fetcher = DatabaseDataFetcher(db_manager=db_manager)

# Register fetchers
# data_fetcher_router.register_fetcher("database", db_data_fetcher)
data_fetcher_router.register_fetcher("s3_wtk", s3_data_fetcher_wtk)
data_fetcher_router.register_fetcher("athena_wtk", athena_data_fetcher_wtk)

VALID_AVG_TYPES = {
    "athena_wtk": {
        "wind_speed": ["global", "yearly", "monthly", "hourly", "none"],
        "production": ["global", "summary", "yearly", "monthly", "all", "none"],
    }
}

# YEARS list for the sample data download feature
SAMPLE_YEARS = {
    "s3_wtk" : [2018, 2019, 2020]
}

# YEARS for which we have wtk data in the S3
ALL_YEARS = {
    "s3_wtk" : list(range(2000,2021))
}
# data_type = "wtk"
VALID_SOURCES = {"athena_wtk", "s3_wtk"}  # <-- new
DEFAULT_SOURCE = "athena_wtk"

# Helper validation functions
def validate_lat(lat: float) -> float:
    if not (-90 <= lat <= 90):
        raise HTTPException(status_code=400, detail="Latitude must be between -90 and 90.")
    return lat

def validate_lng(lng: float) -> float:
    if not (-180 <= lng <= 180):
        raise HTTPException(status_code=400, detail="Longitude must be between -180 and 180.")
    return lng

def validate_height(height: int) -> int:
    if not (0 < height <= 300):
        raise HTTPException(status_code=400, detail="Height must be between 1 and 300 meters.")
    return height

def validate_avg_type(avg_type: str, source: str) -> str:
    allowed = VALID_AVG_TYPES[source]["wind_speed"]
    if avg_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid avg_type. Must be one of: {allowed} for {source}."
        )
    return avg_type

def validate_production_avg_type(avg_type: str, source: str) -> str:
    allowed = VALID_AVG_TYPES[source]["production"]
    if avg_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid time_period. Must be one of: {allowed} for {source}."
        )
    return avg_type

def validate_selected_powercurve(selected_powercurve: str) -> str:
    if not re.match(r'^[\w\-.]+$', selected_powercurve):
        raise HTTPException(status_code=400, detail="Invalid selected_powercurve name.")
    if selected_powercurve not in power_curve_manager.power_curves:
        raise HTTPException(status_code=400, detail="Selected power curve not found.")
    return selected_powercurve

def validate_source(source: str) -> str:
    if source not in VALID_SOURCES:
        raise HTTPException(status_code=400, detail=f"Invalid source for WTK data. Must be one of: {sorted(VALID_SOURCES)}.")
    return source

def validate_year(year: int, source: str) -> int:
    if year not in ALL_YEARS[source]:
        raise HTTPException(status_code=400, detail=f"Invalid year for WTK data. Currently supporting years 2000-2022")
    return year

def validate_n_neighbor(n_neighbor: int) -> int:
    if not 1<=n_neighbor<=4: # have to change the limit if needed later on
        raise HTTPException(status_code=400, detail=f"Invalid number of neighbors. Currently supporting upto 4 nearest neighbors")
    return n_neighbor

def _get_windspeed_core(
    lat: float,
    lng: float,
    height: int,
    avg_type: str,
    source: str
):
    """
    Core function to retrieve wind speed data from the source database.
    Args:
        lat (float): Latitude of the location.
        lng (float): Longitude of the location.
        height (int): Height in meters.
        avg_type (str): Type of average to retrieve. Must be one of: global (default), monthly, yearly.
        source (str): Source of the data. Must be one of: athena_, s3, database.
    """
    lat = validate_lat(lat)
    lng = validate_lng(lng)
    height = validate_height(height)
    source = validate_source(source)
    avg_type = validate_avg_type(avg_type, source)

    params = {
        "lat": lat,
        "lng": lng,
        "height": height,
        "avg_type": avg_type
    }
    data = data_fetcher_router.fetch_data(params, source=source)
    if data is None:
        raise HTTPException(status_code=404, detail="Data not found")
    return data

@router.get(
    "/windspeed/{avg_type}",
    summary="Retrieve wind speed with avg type - wtk data",
    response_model=WindSpeedResponse,
    responses={
        200: {
            "description": "Wind speed data retrieved successfully",
            "model": WindSpeedResponse
        },
        500: {"description": "Internal server error"},
    }
)
def get_windspeed_with_avg_type(
    avg_type: str = Path(..., description="Type of average to retrieve."),
    lat: float = Query(..., description="Latitude of the location."),
    lng: float = Query(..., description="Longitude of the location."),
    height: int = Query(..., description="Height in meters."),
    source: str = Query(DEFAULT_SOURCE, description="Source of the data.")
):
    try:
        return _get_windspeed_core(lat, lng, height, avg_type, source)
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error.")

@router.get(
    "/windspeed",
    summary="Retrieve wind speed with default global avg - wtk data",
    response_model=WindSpeedResponse,
    responses={
        200: {
            "description": "Wind speed data retrieved successfully",
            "model": WindSpeedResponse
        },
        500: {"description": "Internal server error"},
    }
)
def get_windspeed(
    lat: float = Query(..., description="Latitude of the location."),
    lng: float = Query(..., description="Longitude of the location."),
    height: int = Query(..., description="Height in meters."),
    source: str = Query(DEFAULT_SOURCE, description="Source of the data.")
):
    try:
        return _get_windspeed_core(lat, lng, height, "global", source)
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error.")

@router.get(
        "/available-powercurves",
        summary="Fetch all available power curves",
        response_model=AvailablePowerCurvesResponse,
        responses={
            200: {
                "description": "Available power curves retrieved successfully",
                "model": AvailablePowerCurvesResponse
            },
            500: {"description": "Internal server error"},
        }
)
def fetch_available_powercurves():
    try:
        all_curves = list(power_curve_manager.power_curves.keys())
        def extract_kw(curve_name: str):
            import re
            match = re.search(r"nrel-reference-([0-9.]+)kW", curve_name)
            if match:
                return float(match.group(1))
            return float('inf')
        nrel_curves = [c for c in all_curves if c.startswith("nrel-reference-")]
        other_curves = [c for c in all_curves if not c.startswith("nrel-reference-")]
        nrel_curves_sorted = sorted(nrel_curves, key=extract_kw)
        other_curves_sorted = sorted(other_curves)
        ordered_curves = nrel_curves_sorted + other_curves_sorted
        return {'available_power_curves': ordered_curves}
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error.")

def _get_energy_production_core(
    lat: float,
    lng: float,
    height: int,
    selected_powercurve: str,
    time_period: str,
    source: str
):
    
    """
    Fetches the global, yearly and monthly energy production and average windspeed for a given location, height, and power curve.
    Args:
        lat (float): Latitude of the location.
        lng (float): Longitude of the location.
        height (int): Height in meters.
        time_period (str, optional): Time period to retrieve. Must be one of: global, yearly, monthly, all.
        source (str): Source of the data. Must be one of: athena, s3, database.
    Returns:
        A JSON object containing average windspeeds and energy production at specified time period or global energy production when time period is not specified.
    """
    lat = validate_lat(lat)
    lng = validate_lng(lng)
    height = validate_height(height)
    selected_powercurve = validate_selected_powercurve(selected_powercurve)
    source = validate_source(source)
    time_period = validate_production_avg_type(time_period, source)

    params = {
        "lat": lat,
        "lng": lng,
        "height": height,
        "avg_type": "none"
    }
    df = data_fetcher_router.fetch_data(params, source=source)
    if df is None:
        raise HTTPException(status_code=404, detail="Data not found")

    if time_period == 'global':
        summary_avg_energy_production = power_curve_manager.fetch_avg_energy_production_summary(df, height, selected_powercurve)
        return {"energy_production": summary_avg_energy_production['Average year']['kWh produced']}
    
    elif time_period == 'summary':
        summary_avg_energy_production = power_curve_manager.fetch_avg_energy_production_summary(df, height, selected_powercurve)
        return {"summary_avg_energy_production": summary_avg_energy_production}
    
    elif time_period == 'yearly':
        yearly_avg_energy_production = power_curve_manager.fetch_yearly_avg_energy_production(df, height, selected_powercurve)
        return {"yearly_avg_energy_production": yearly_avg_energy_production}
    
    elif time_period == 'all':
        summary_avg_energy_production = power_curve_manager.fetch_avg_energy_production_summary(df, height, selected_powercurve)
        yearly_avg_energy_production = power_curve_manager.fetch_yearly_avg_energy_production(df, height, selected_powercurve)
        return {
            "energy_production": summary_avg_energy_production['Average year']['kWh produced'],
            "summary_avg_energy_production": summary_avg_energy_production,
            "yearly_avg_energy_production": yearly_avg_energy_production
        }

@router.get(
        "/energy-production/{time_period}",
        summary="Get yearly and monthly energy production estimate and average windspeed for a location at a height with a selected power curve",
        response_model=EnergyProductionResponse,
        responses={
            200: {
                "description": "Energy production data retrieved successfully",
                "model": EnergyProductionResponse
            },
            500: {"description": "Internal server error"},
        }
)
def energy_production_with_period(
    time_period: str = Path(..., description="Time period for production estimate."),lat: float = Query(..., description="Latitude of the location."),
    lng: float = Query(..., description="Longitude of the location."),
    height: int = Query(..., description="Height in meters."),
    selected_powercurve: str = Query(..., description="Selected power curve name."),
    source: str = Query(DEFAULT_SOURCE, description="Source of the data.")
):
    try:
        return _get_energy_production_core(lat, lng, height, selected_powercurve, time_period, source)
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error.")

@router.get(
    "/energy-production",
    summary="Get global energy production estimate for a location at a height with a selected power curve",
    response_model=EnergyProductionResponse,
    responses={
        200: {
            "description": "Energy production data retrieved successfully",
            "model": EnergyProductionResponse
        },
        500: {"description": "Internal server error"},
    }
)
def energy_production(
    lat: float = Query(..., description="Latitude of the location."),
    lng: float = Query(..., description="Longitude of the location."),
    height: int = Query(..., description="Height in meters."),
    selected_powercurve: str = Query(..., description="Selected power curve name."),
    source: str = Query(DEFAULT_SOURCE, description="Source of the data.")
):
    try:
        return _get_energy_production_core(lat, lng, height, selected_powercurve, "all", source)
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error.")

def _download_csv_core(
    gridIndices: List[str],
    years: List[int],
    source: str
):
    source = validate_source(source)
    years= [validate_year(year,source) for year in years]
    
    params = {
        "gridIndices": gridIndices,
        "years": years
    }

    df = data_fetcher_router.fetch_data(params, source=source)

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No data found for the specified parameters")
    
    return df

    
@router.get(
    "/download-csv",
    summary="Download csv file for windspeed for a specific location for certain year(s) with 1 neighbor"
)
def download_csv(
    gridIndex: str = Query(..., description="Grid index with respect to user selected coordinate"),
    years: List[int] = Query(SAMPLE_YEARS["s3_wtk"], description="years of which the data to download"),
    source: str = Query("s3_wtk", description="Source of the data.")
):
    try:
        # Getting DataFrame from core function
        df = _download_csv_core([gridIndex], years, source)
        
        # Converting DataFrame to CSV
        csv_io = io.StringIO()
        df.to_csv(csv_io, index=False)
        csv_io.seek(0)
        
        return StreamingResponse(
            iter([csv_io.getvalue()]),
            media_type="text/csv; charset=utf-8"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post(
    "/download-csv-batch",
    summary="Download multiple CSVs (one per neighbor) as a streamed ZIP",
)
def download_csv_batch(
    payload: NearestLocationsResponse,
    years: List[int] = Query(SAMPLE_YEARS["s3_wtk"], description="years of which the data to download"),
    source: str = Query("s3_wtk", description="Source of the data."),
):
    try:
        # Spooled file: stays in memory until threshold, then spills to disk automatically
        spooled = tempfile.SpooledTemporaryFile(max_size=30 * 1024 * 1024, mode="w+b")  # 30MB threshold

        with zipfile.ZipFile(spooled, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for loc in payload.locations:
                df = _download_csv_core([loc.index], years, source)
                csv_io = io.StringIO()
                df.to_csv(csv_io, index=False)
                csv_io.seek(0)
                file_name = f"wind_data_{format_coordinate(loc.latitude)}_{format_coordinate(loc.longitude)}.csv"
                zf.writestr(file_name, csv_io.getvalue())

        spooled.seek(0)

        headers = {
            "Content-Disposition": f'attachment; filename="wind_data_{len(payload.locations)}_points.zip"'
        }

        return StreamingResponse(chunker(spooled), media_type="application/zip", headers=headers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get(
    "/nearest-locations",
    summary="Find nearest grid locations",
    response_model=NearestLocationsResponse,
    responses={
        200: {"description": "Nearest locations retrieved successfully"},
        400: {"description": "Bad request"},
        500: {"description": "Internal server error"},
    },
)
def nearest_locations(
    lat: float = Query(..., description="Latitude of the target location."),
    lng: float = Query(..., description="Longitude of the target location."),
    n_neighbors: int = Query(1, description="Number of nearest grid points.", ge=1, le=4),
    source: str = Query(DEFAULT_SOURCE, description=f"Source of the data"),
):
    try:
        lat = validate_lat(lat)
        lng = validate_lng(lng)
        n_neighbors = validate_n_neighbor(n_neighbors)
        source = validate_source(source)

        grid_lookup_map = {
            "athena_wtk": athena_data_fetcher_wtk,
        }
        
        fetcher = grid_lookup_map.get(source)
        if not fetcher:
            raise HTTPException(
                status_code=400, 
                detail=f"Grid lookup not available for source='{source}'"
            )

        # Call find_nearest_locations directly on the Athena fetcher
        result = fetcher.find_nearest_locations(lat=lat, lng=lng, n_neighbors=n_neighbors)
        
        locations = [
            {
                "index": str(i), 
                "latitude": float(a), 
                "longitude": float(o)
            } 
            for i, a, o in result
        ]

        return {"locations": locations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")