"""
Model configuration for WindWatts API.

Defines the configuration for all supported wind data models including
data sources, valid parameters, and model-specific settings.

"""

MODEL_CONFIG = {
    "era5": {
        "sources": ["athena", "s3"],
        "default_source": "athena",
        "period_type": {
            "windspeed": ["all", "annual"],
            "production": ["all", "summary", "annual"]
        },
        "years": {
            "full": list(range(2013, 2024)),
            "sample": [2020, 2021, 2022, 2023]
        },
        "heights": [30, 40, 50, 60, 80, 100]
    },
    "wtk": {
        "sources": ["athena", "s3"],
        "default_source": "athena",
        "period_type": {
            "windspeed": ["all", "annual", "monthly", "hourly"],
            "production": ["all", "summary", "annual", "monthly"]
        },
        "years": {
            "full": list(range(2000, 2021)),
            "sample": [2018, 2019, 2020]
        },
        "heights": [40, 60, 80, 100, 120, 140, 160, 200]
    },
    "ensemble": {
        "sources": ["athena"],
        "default_source": "athena",
        "period_type": {
            "windspeed": ["all"],
            "production": ["all"]
        },
        "years": {
            "full": list(range(2013, 2024)),
            "sample": []
        },
        "heights": [30, 40, 50, 60, 80, 100]
    }
}
