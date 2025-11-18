from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field

Numeric = Union[int, float]
AlphaNumeric = Union[str, Numeric]
AlphaNumericNone = Union[str, Numeric, None]
ValueMapNumeric = Dict[str, Numeric]
ValueMapAlphaNumeric = Dict[str, AlphaNumeric]
ValueMapAlphaNumericNone = Dict[str, AlphaNumericNone]
ValueMapNumericList = List[ValueMapNumeric]

# Wind speed response models for different avg_types
class GlobalWindSpeedResponse(BaseModel):
    global_avg: float

    model_config = {
        "json_schema_extra": {
            "example": {"global_avg": 2.1}
        }
    }

class YearlyWindSpeedResponse(BaseModel):
    yearly_avg: ValueMapNumericList

    model_config = {
        "json_schema_extra": {
            "example": {
                "yearly_avg": [
                    {"year": 2020, "windspeed_100m": 5.23},
                    {"year": 2021, "windspeed_100m": 5.34}
                ]
            }
        }
    }

class MonthlyWindSpeedResponse(BaseModel):
    monthly_avg: ValueMapNumericList

    model_config = {
        "json_schema_extra": {
            "example": {
                "monthly_avg": [
                    {"month": 1, "windspeed_100m": 5.12},
                    {"month": 2, "windspeed_100m": 5.45},
                    {"month": 12, "windspeed_100m": 6.10}
                ]
            }
        }
    }

# Union type for wind speed responses - FastAPI will show all examples
WindSpeedResponse = Union[GlobalWindSpeedResponse, YearlyWindSpeedResponse, MonthlyWindSpeedResponse]

class AvailablePowerCurvesResponse(BaseModel):
    available_power_curves: List[str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "available_power_curves": [
                    "nrel-reference-2.5kW",
                    "nrel-reference-100kW",
                ]
            }
        }
    }

# Energy production response models for different time_periods
class SummaryEnergyProductionResponse(BaseModel):
    energy_production: Numeric = Field(description="global-averaged kWh produced")

    model_config = {
        "json_schema_extra": {
            "example": {"energy_production": 12345.67}
        }
    }

class YearlyEnergyProductionResponse(BaseModel):
    yearly_avg_energy_production: Dict[str, ValueMapAlphaNumeric]

    model_config = {
        "json_schema_extra": {
            "example": {
                "yearly_avg_energy_production": {
                    "2001": {"Average wind speed (m/s)": "5.65", "kWh produced": 250117},
                    "2002": {"Average wind speed (m/s)": "5.72", "kWh produced": 264044}
                }
            }
        }
    }

class AllEnergyProductionResponse(BaseModel):
    energy_production: Numeric = Field(description="global-averaged kWh produced")
    summary_avg_energy_production: Dict[str, ValueMapAlphaNumericNone]
    yearly_avg_energy_production: Dict[str, ValueMapAlphaNumeric]

    model_config = {
        "json_schema_extra": {
            "example": {
                "energy_production": 500,
                "summary_avg_energy_production": {
                    "Lowest year": {"year": 2015, "Average wind speed (m/s)": "5.36", "kWh produced": 202791},
                    "Average year": {"year": None, "Average wind speed (m/s)": "5.86", "kWh produced": 267712},
                    "Highest year": {"year": 2014, "Average wind speed (m/s)": "6.32", "kWh produced": 326354}
                },
                "yearly_avg_energy_production": {
                    "2001": {"Average wind speed (m/s)": "5.65", "kWh produced": 250117}
                }
            }
        }
    }

class MonthlyEnergyProductionResponse(BaseModel):
    monthly_avg_energy_production: Dict[str, ValueMapAlphaNumeric]

    model_config = {
        "json_schema_extra": {
            "example": {
                "monthly_avg_energy_production": {
                    "Jan": {"Average wind speed, m/s": "3.80", "kWh produced": "5,934"},
                    "Feb": {"Average wind speed, m/s": "3.92", "kWh produced": "6,357"}
                }
            }
        }
    }

# Union type for energy production responses
EnergyProductionResponse = Union[SummaryEnergyProductionResponse, YearlyEnergyProductionResponse, AllEnergyProductionResponse, MonthlyEnergyProductionResponse]

class HealthCheckResponse(BaseModel):
    status: Literal["up"] = "up"

    model_config = {
        "json_schema_extra": {
            "example": {"status": "up"}
        }
    }

class GridLocation(BaseModel):
    index: str = Field(..., description="Grid point identifier/index")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")

    model_config = {
        "json_schema_extra": {
            "example": {"index": "031233", "latitude": 43.653, "longitude": -79.47437700534891}
        }
    }

class NearestLocationsResponse(BaseModel):
    locations: List[GridLocation] = Field(
        ..., 
        min_length=1, 
        max_length=4, 
        description="List of nearest grid locations (1-4 points)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "locations": [
                    {"index": "031233", "latitude": 43.653, "longitude": -79.47437700534891},
                    {"index": "031234", "latitude": 43.653, "longitude": -79.22437433155213}
                ]
            }
        }
    }