"""
Validation functions for WindWatts API.
"""

import re
from fastapi import HTTPException

from app.config.model_config import MODEL_CONFIG
from app.power_curve.global_power_curve_manager import power_curve_manager


def validate_model(model: str) -> str:
    """Validate model parameter"""
    if model not in MODEL_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Must be one of: {list(MODEL_CONFIG.keys())}",
        )
    return model


def validate_source(model: str, source: str) -> str:
    """Validate source for given model"""
    valid_sources = MODEL_CONFIG[model]["sources"]
    if source not in valid_sources:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source for {model}. Must be one of: {valid_sources}",
        )
    return source


def validate_period_type(model: str, period_type: str, data_type: str) -> str:
    """Validate period parameter for given model and data type"""
    valid_periods = MODEL_CONFIG[model]["period_type"].get(data_type, [])
    if period_type not in valid_periods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid period for {model}. Must be one of: {valid_periods}",
        )
    return period_type


def validate_lat(model: str, lat: float) -> float:
    """Validate latitude parameter"""
    min_lat = MODEL_CONFIG[model]["grid_info"].get("min_lat")
    max_lat = MODEL_CONFIG[model]["grid_info"].get("max_lat")
    if not (min_lat <= lat <= max_lat):
        raise HTTPException(
            status_code=400,
            detail=f"Latitude for {model} must be between {min_lat} and {max_lat}.",
        )
    return lat


def validate_lng(model: str, lng: float) -> float:
    """Validate longitude parameter"""
    min_lng = MODEL_CONFIG[model]["grid_info"].get("min_long")
    max_lng = MODEL_CONFIG[model]["grid_info"].get("max_long")
    if not (min_lng <= lng <= max_lng):
        raise HTTPException(
            status_code=400,
            detail=f"Longitude for {model} must be between {min_lng} and {max_lng}.",
        )
    return lng


def validate_height(model: str, height: int) -> int:
    """Validate height parameter"""
    valid_heights = MODEL_CONFIG[model].get("heights", [])
    if valid_heights and height not in valid_heights:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid height for {model}. Must be one of: {valid_heights}",
        )
    return height


def validate_powercurve(powercurve: str) -> str:
    """Validate power curve name"""
    if not re.match(r"^[\w\-.]+$", powercurve):
        raise HTTPException(status_code=400, detail="Invalid power curve name.")
    if powercurve not in power_curve_manager.power_curves:
        raise HTTPException(status_code=400, detail="Power curve not found.")
    return powercurve


def validate_year(year: int, model: str) -> int:
    """Validate year for given model"""
    valid_years = MODEL_CONFIG[model]["years"].get("full", [])
    if valid_years and year not in valid_years:
        year_range = f"{min(valid_years)}-{max(valid_years)}"
        raise HTTPException(
            status_code=400,
            detail=f"Invalid year for {model}. Currently supporting years {year_range}",
        )
    return year


def validate_limit(limit: int) -> int:
    """Validate limit parameter for grid points"""
    if not 1 <= limit <= 4:
        raise HTTPException(
            status_code=400,
            detail="Invalid limit. Currently supporting up to 4 nearest grid points",
        )
    return limit
