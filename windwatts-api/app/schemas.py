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

class HourlyWindSpeedResponse(BaseModel):
    hourly_avg: ValueMapNumericList

    model_config = {
        "json_schema_extra": {
            "example": {
                "hourly_avg": [
                    {"hour": 0, "windspeed_100m": 5.12},
                    {"hour": 2, "windspeed_100m": 5.45},
                    {"hour": 10, "windspeed_100m": 6.10}
                ]
            }
        }
    }

# Union type for wind speed responses - FastAPI will show all examples
WindSpeedResponse = Union[GlobalWindSpeedResponse, YearlyWindSpeedResponse, MonthlyWindSpeedResponse, HourlyWindSpeedResponse]

class AvailablePowerCurvesResponse(BaseModel):
    available_power_curves: List[str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "available_power_curves": [
                    "nlr-reference-2.5kW",
                    "nlr-reference-100kW",
                ]
            }
        }
    }

# Energy production response models for different time_periods
class AllEnergyProductionResponse(BaseModel):
    energy_production: Numeric = Field(description="global-averaged kWh produced")

    model_config = {
        "json_schema_extra": {
            "example": {"energy_production": 12345.67}
        }
    }

class SummaryEnergyProductionResponse(BaseModel):
    summary_avg_energy_production: Dict[str, ValueMapAlphaNumericNone]

    model_config = {
        "json_schema_extra": {
            "example": {
                "summary_avg_energy_production": {
                    "Lowest year": {"year": 2015, "Average wind speed (m/s)": "5.36", "kWh produced": 202791},
                    "Average year": {"year": None, "Average wind speed (m/s)": "5.86", "kWh produced": 267712},
                    "Highest year": {"year": 2014, "Average wind speed (m/s)": "6.32", "kWh produced": 326354}
                }
            }
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

class FullEnergyProductionResponse(BaseModel):
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
EnergyProductionResponse = Union[AllEnergyProductionResponse, SummaryEnergyProductionResponse, YearlyEnergyProductionResponse, FullEnergyProductionResponse, MonthlyEnergyProductionResponse]

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

class TimeseriesBatchRequest(BaseModel):
    locations: List[GridLocation] = Field(
        ..., 
        min_length=1,
        description="List of grid locations to download timeseries data for"
    )
    years: Optional[List[int]] = Field(
        None,
        description="Years to download (defaults to sample years if not provided)"
    )
    source: str = Field(
        "s3",
        description="Data source: athena or s3 (typically s3 for timeseries downloads)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "locations": [
                    {"index": "031233", "latitude": 43.653, "longitude": -79.47437700534891},
                    {"index": "031234", "latitude": 43.653, "longitude": -79.22437433155213}
                ],
                "years": [2020, 2021, 2022],
                "source": "s3"
            }
        }
    }

class ModelInfoResponse(BaseModel):
    model: str = Field(..., description="Data model name")
    supported_periods: Dict[str, List[str]] = Field(
        default_factory=dict, 
        description="Supported aggregation periods for windspeed/ production"
    )
    available_years: List[int] = Field(
        ..., 
        description="Available years for timeseries data"
    )
    available_heights: List[int] = Field(
        ...,
        description="Supported hub heights (in meters)"
    )
    grid_info: Dict[str, AlphaNumeric] = Field(
        default_factory=dict,
        description="Metadata about the model grid (bounds, resolution, etc.)"
    )
    references: List[str] = Field(
        ...,
        description="References of relevant publications or documents"
    )
    links: List[str] = Field(
        ...,
        description="Links to data sources or relevant resources"
    )
    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "era5",
                "supported_periods": {
                    "windspeed": ["all", "annual"],
                    "production": ["all", "summary", "annual", "full"]
                },
                "available_years":[2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
                "available_heights": [30, 40, 50, 60, 80, 100],
                "grid_info": {
                    "min_lat": 23.402,
                    "min_long": -137.725,
                    "max_lat": 51.403,
                    "max_long": -44.224,
                    "spatial_resolution": "31 km",
                    "temporal_resolution": "1 hour"
                },
                "links": [
                    "https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5"
                ],
                "references": [
                    'Phillips, C., L. M. Sheridan, P. Conry, D. K. Fytanidis, D. Duplyakin, S. Zisman, N. Duboc, M. Nelson, R. Kotamarthi, R. Linn, M. Broersma, T. Spijkerboer, and H. Tinnesand. 2022. "Evaluation of Obstacle Modelling Approaches for Resource Assessment and Small Wind Turbine Siting: Case Study in the Northern Netherlands." Wind Energy Science 7: 1153-1169. https://doi.org/10.5194/wes-7-1153-2022'
                ]
            }
        }
    }