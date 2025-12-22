"""
Core data retrieval functions for WindWatts API.

Provides the core business logic for fetching wind speed, energy production,
and timeseries data from various data sources.
"""

from typing import List
from fastapi import HTTPException
from app.data_fetchers.data_fetcher_router import DataFetcherRouter

from app.utils.validation import (
    validate_lat,
    validate_lng,
    validate_height,
    validate_model,
    validate_source,
    validate_period_type,
    validate_powercurve,
    validate_year
)
from app.power_curve.global_power_curve_manager import power_curve_manager


def get_windspeed_core(
    model: str,
    lat: float,
    lng: float,
    height: int,
    period: str,
    source: str,
    data_fetcher_router: DataFetcherRouter
):
    """
    Core function to retrieve wind speed data from the source database.
    
    Args:
        model (str): Data model (era5, wtk, ensemble).
        lat (float): Latitude of the location.
        lng (float): Longitude of the location.
        height (int): Height in meters.
        period (str): Type of period to retrieve (all, annual, monthly, hourly, none).
        source (str): Source of the data (athena, s3).
        data_fetcher_router: Router instance for fetching data.

    Returns:
        Wind speed data from the data source.
    """
    lat = validate_lat(model, lat)
    lng = validate_lng(model, lng)
    model = validate_model(model)
    height = validate_height(model, height)
    source = validate_source(model, source)
    period = validate_period_type(model, period, "windspeed")

    params = {
        "lat": lat,
        "lng": lng,
        "height": height,
        "period": period
    }
    
    key = f"{source}_{model}"
    data = data_fetcher_router.fetch_data(params, key=key)
    if data is None:
        raise HTTPException(status_code=404, detail="Data not found")
    return data


def get_production_core(
    model: str,
    lat: float,
    lng: float,
    height: int,
    powercurve: str,
    period: str,
    source: str,
    data_fetcher_router: DataFetcherRouter
):
    """
    Core function to retrieve energy production data.
    
    Args:
        model (str): Data model (era5, wtk, ensemble).
        lat (float): Latitude of the location.
        lng (float): Longitude of the location.
        height (int): Height in meters.
        powercurve (str): Power curve name.
        period (str): Time period to retrieve (all, summary, annual, monthly).
        source (str): Source of the data.
        data_fetcher_router: Router instance for fetching data.

    Returns:
        A dictionary containing energy production data.
    """
    lat = validate_lat(model, lat)
    lng = validate_lng(model, lng)
    model = validate_model(model)
    height = validate_height(model, height)
    powercurve = validate_powercurve(powercurve)
    source = validate_source(model, source)
    period = validate_period_type(model, period, "production")

    # Always fetch raw data for production calculations
    params = {
        "lat": lat,
        "lng": lng,
        "height": height
    }
    
    key = f"{source}_{model}"
    df = data_fetcher_router.fetch_raw(params, key=key)
    if df is None:
        raise HTTPException(status_code=404, detail="Data not found")
    
    if period == 'all':
        summary_avg_energy_production = power_curve_manager.calculate_energy_production_summary(df, height, powercurve)
        return {"energy_production": summary_avg_energy_production['Average year']['kWh produced']}
    
    elif period == 'summary':
        summary_avg_energy_production = power_curve_manager.calculate_energy_production_summary(df, height, powercurve)
        return {"summary_avg_energy_production": summary_avg_energy_production}
    
    elif period == 'annual':
        yearly_avg_energy_production = power_curve_manager.calculate_yearly_energy_production(df, height, powercurve)
        return {"yearly_avg_energy_production": yearly_avg_energy_production}
    
    elif period == 'monthly':
        monthly_avg_energy_production = power_curve_manager.calculate_monthly_energy_production(df, height, powercurve)
        return {"monthly_avg_energy_production": monthly_avg_energy_production}
    
    elif period == 'full':
        summary_avg_energy_production = power_curve_manager.calculate_energy_production_summary(df, height, powercurve)
        yearly_avg_energy_production = power_curve_manager.calculate_yearly_energy_production(df, height, powercurve)
        return {
            "energy_production": summary_avg_energy_production['Average year']['kWh produced'],
            "summary_avg_energy_production": summary_avg_energy_production,
            "yearly_avg_energy_production": yearly_avg_energy_production
        }        

def get_timeseries_core(
    model: str,
    gridIndices: List[str],
    years: List[int],
    source: str,
    data_fetcher_router: DataFetcherRouter
):
    """
    Core function to retrieve timeseries data for download.
    
    Args:
        model (str): Data model (era5, wtk, ensemble).
        gridIndices (List[str]): List of grid indices to retrieve.
        years (List[int]): List of years to retrieve, default to sample years.
        source (str): Source of the data.
        data_fetcher_router: Router instance for fetching data.
        
    Returns:
        str: CSV content as string.
    """
    from io import StringIO
    from app.config.model_config import MODEL_CONFIG
    
    model = validate_model(model)
    source = validate_source(model, source)
    
    if years is None:
        years = MODEL_CONFIG[model]["years"].get("sample", [])
    
    years = [validate_year(year, model) for year in years]
    
    params = {
        "gridIndices": gridIndices,
        "years": years
    }
    
    key = f"{source}_{model}"
    df = data_fetcher_router.fetch_data(params, key=key)

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No data found for the specified parameters")
    
    # Convert DataFrame to CSV string
    csv_io = StringIO()
    df.to_csv(csv_io, index=False)
    return csv_io.getvalue()
