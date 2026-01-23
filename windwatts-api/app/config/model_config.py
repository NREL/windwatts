"""
Model configuration for WindWatts API.

Defines the configuration for all supported wind data models including
data sources, valid parameters, and model-specific settings.

"""
TEMPORAL_SCHEMAS = {
    "full_hourly": {
        "description": "Full 8760-hour timeseries with datetime index",
        "column_config": {
            "time_column": "time"
        },
        "validation": {
            "required_columns": ["time"]
        },
        "use_swi": False
    },
    "aggregated_mohr": {
        "description": "Hourly-aggregated (12*24) with month-hour encoding",
        "column_config": {
            "time_column": "mohr"
        },
        "validation": {
            "required_columns": ["mohr", "year"],
        },
        "use_swi": False
    },
    "quantile_yearly": {
        "description": "Quantile distributions per year",
        "column_config": {
            "probability_column": "probability"
        },
        "validation": {
            "required_columns": ["probability", "year"]
        },
        "use_swi": True
    },
    "quantile_atemporal": {
        "description": "Global quantiles without temporal dimension",
        "column_config": {
            "probability_column": "probability",
            "no_temporal_dims": True
        },
        "validation": {
            "required_columns": ["probability"],
            "no_year_column": True
        },
        "use_swi": False
    }
}

MODEL_CONFIG = {
    "era5-quantiles": {
        "source": "athena",
        "schema": "quantile_yearly",
        "period_type": {
            "windspeed": ["all", "annual"],
            "production": ["all", "summary", "annual", "full"],
        },
        "years": {"full": list(range(2013, 2024)), "sample": [2020, 2021, 2022, 2023]},
        "heights": [30, 40, 50, 60, 80, 100],
        "grid_info": {
            "min_lat": 23.402,
            "min_long": -137.725,
            "max_lat": 51.403,
            "max_long": -44.224,
            "spatial_resolution": "31 km",
            "temporal_resolution": "1 hour",
        },
        "links": ["https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5"],
        "references": [
            'Phillips, C., L. M. Sheridan, P. Conry, D. K. Fytanidis, D. Duplyakin, S. Zisman, N. Duboc, M. Nelson, R. Kotamarthi, R. Linn, M. Broersma, T. Spijkerboer, and H. Tinnesand. 2022. "Evaluation of Obstacle Modelling Approaches for Resource Assessment and Small Wind Turbine Siting: Case Study in the Northern Netherlands." Wind Energy Science 7: 1153-1169. https://doi.org/10.5194/wes-7-1153-2022'
        ],
    },
    "wtk-timeseries": {
        "source": "athena",
        "schema": "aggregated_mohr",
        "period_type": {
            "windspeed": ["all", "annual", "monthly", "hourly"],
            "production": ["all", "summary", "annual", "monthly", "full"],
            "timeseries_export": ["hourly", "monthly"]
        },
        "years": {"full": list(range(2000, 2021)), "sample": [2018, 2019, 2020]},
        "heights": [40, 60, 80, 100, 120, 140, 160, 200],
        "grid_info": {
            "min_lat": 7.75129,
            "min_long": -179.99918,
            "max_lat": 78.392685,
            "max_long": 180.0,
            "spatial_resolution": "2 km",
            "temporal_resolution": "1 hour",
        },
        "links": ["https://www.nrel.gov/grid/wind-toolkit"],
        "references": [
            "Draxl, C., B.M. Hodge, A. Clifton, and J. McCaa. 2015. Overview and Meteorological Validation of the Wind Integration National Dataset Toolkit (Technical Report, NREL/TP-5000-61740). Golden, CO: National Laboratory of the Rockies",
            'Draxl, C., B.M. Hodge, A. Clifton, and J. McCaa. 2015. "The Wind Integration National Dataset (WIND) Toolkit." Applied Energy 151: 355366',
            "King, J., A. Clifton, and B.M. Hodge. 2014. Validation of Power Output for the WIND Toolkit (Technical Report, NREL/TP-5D00-61714). Golden, CO: National Laboratory of the Rockies",
        ],
    },
    "ensemble-quantiles": {
        "source": "athena",
        "schema": "quantile_atemporal",
        "period_type": {"windspeed": ["all"], "production": ["all"]},
        "years": {"full": list(range(2013, 2024)), "sample": []},
        "heights": [30, 40, 50, 60, 80, 100],
        "grid_info": {
            "min_lat": 23.402,
            "min_long": -137.725,
            "max_lat": 51.403,
            "max_long": -44.224,
            "spatial_resolution": "31 km",
            "temporal_resolution": "1 hour",
        },
        "links": [],
        "references": [
            "Kevin Menear, Sameer Shaik, Lindsay Sheridan, Dmitry Duplyakin, and Caleb Phillips. Methods for High-Accuracy Wind Resource Assessment to Support Distributed Wind Turbine Siting. Under Review."
        ],
    },
    "era5-timeseries": {
        "source": "s3",
        "schema": "full_hourly",
        "period_type": {
            "timeseries_export": ["hourly", "monthly"] 
        },
        "years": {"full": list(range(2013, 2024)), "sample": [2020, 2021, 2022, 2023]},
        "heights": [30, 40, 50, 60, 80, 100],
        "grid_info": {
            "min_lat": 23.402,
            "min_long": -137.725,
            "max_lat": 51.403,
            "max_long": -44.224,
            "spatial_resolution": "31 km",
            "temporal_resolution": "1 hour",
        },
        "links": ["https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5"],
        "references": [
            'Phillips, C., L. M. Sheridan, P. Conry, D. K. Fytanidis, D. Duplyakin, S. Zisman, N. Duboc, M. Nelson, R. Kotamarthi, R. Linn, M. Broersma, T. Spijkerboer, and H. Tinnesand. 2022. "Evaluation of Obstacle Modelling Approaches for Resource Assessment and Small Wind Turbine Siting: Case Study in the Northern Netherlands." Wind Energy Science 7: 1153-1169. https://doi.org/10.5194/wes-7-1153-2022'
        ],
    },
}
