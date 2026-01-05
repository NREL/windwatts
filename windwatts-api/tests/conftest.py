"""
Pytest configuration and fixtures for mocking external AWS dependencies.
"""

import os
import pandas as pd
from unittest.mock import MagicMock, patch


# Global patch objects that need to persist
_boto3_patch = None
_mock_clients = {}
_windwatts_patches = []


def pytest_configure(config):
    """
    Pytest hook that runs very early, before any test collection.
    Patch windwatts_data clients and boto3 to prevent real AWS calls during module imports.
    """
    global _boto3_patch, _mock_clients, _windwatts_patches

    # Add ensemble configuration to environment variables
    os.environ["SOURCES_ENSEMBLE_BUCKET_NAME"] = "windwatts-era5"
    os.environ["SOURCES_ENSEMBLE_ATHENA_TABLE_NAME"] = "ensemble_101"
    os.environ["SOURCES_ENSEMBLE_ALT_ATHENA_TABLE_NAME"] = ""

    # Mock the windwatts_data client classes to return realistic data
    def create_mock_windwatts_client():
        """Create a mock windwatts client that returns realistic wind data"""
        mock_client = MagicMock()

        # Mock different fetch methods that return data in the format the API expects
        mock_client.fetch_global_avg_at_height = MagicMock(
            return_value={"global_avg": 8.60}
        )

        mock_client.fetch_yearly_avg_at_height = MagicMock(
            return_value={
                "yearly_avg": [
                    {"year": 2013, "windspeed_40m": 8.93},
                    {"year": 2014, "windspeed_40m": 8.61},
                    {"year": 2015, "windspeed_40m": 8.34},
                    {"year": 2016, "windspeed_40m": 8.72},
                    {"year": 2017, "windspeed_40m": 8.85},
                ]
            }
        )

        mock_client.fetch_monthly_avg_at_height = MagicMock(
            return_value={
                "monthly_avg": [
                    {"month": 1, "windspeed_40m": 8.5},
                    {"month": 2, "windspeed_40m": 8.7},
                ]
            }
        )

        mock_client.fetch_hourly_avg_at_height = MagicMock(
            return_value={
                "hourly_avg": [
                    {"hour": 0, "windspeed_40m": 8.5},
                    {"hour": 1, "windspeed_40m": 8.6},
                ]
            }
        )

        # Mock production/energy calculation methods
        mock_client.calculate_global_energy = MagicMock(return_value=539072)

        mock_client.calculate_summary_energy = MagicMock(
            return_value={
                "Lowest year": {
                    "year": 2023.0,
                    "Average wind speed (m/s)": "8.12",
                    "kWh produced": 504212,
                },
                "Average year": {
                    "year": None,
                    "Average wind speed (m/s)": "8.60",
                    "kWh produced": 539072,
                },
                "Highest year": {
                    "year": 2013.0,
                    "Average wind speed (m/s)": "8.93",
                    "kWh produced": 579146,
                },
            }
        )

        mock_client.calculate_yearly_energy = MagicMock(
            return_value={
                "2013": {"Average wind speed (m/s)": "8.93", "kWh produced": 579146},
                "2014": {"Average wind speed (m/s)": "8.61", "kWh produced": 552516},
                "2015": {"Average wind speed (m/s)": "8.34", "kWh produced": 521722},
            }
        )

        mock_client.calculate_monthly_energy = MagicMock(
            return_value={
                "Jan": {"Average wind speed, m/s": "8.5", "kWh produced": "45000"},
                "Feb": {"Average wind speed, m/s": "8.7", "kWh produced": "43000"},
            }
        )

        # Mock the query_athena method for raw queries
        def mock_query_athena(query, convert_to_dataframe=True, *args, **kwargs):
            if convert_to_dataframe:
                return pd.DataFrame(
                    {
                        "wind_speed_40m": [8.93, 8.61, 8.34, 8.72, 8.85],
                        "wind_speed_80m": [9.5, 9.2, 8.9, 9.3, 9.4],
                        "year": [2013, 2014, 2015, 2016, 2017],
                        "month": [1, 2, 3, 4, 5],
                    }
                )
            else:
                return [
                    [
                        "wind_speed_30m",
                        "wind_speed_40m",
                        "wind_speed_50m",
                        "wind_speed_60m",
                        "wind_speed_80m",
                        "wind_speed_100m",
                    ]
                ]

        mock_client.query_athena = mock_query_athena

        # Mock fetch_df for raw data fetches (used by production endpoints)
        def mock_fetch_df(lat, long, height, *args, **kwargs):
            # Return DataFrame with windspeed column named according to height
            # Also includes timestamp, year, month, hour, mohr for power curve calculations
            timestamps = pd.date_range("2020-01-01", periods=5, freq="h")
            return pd.DataFrame(
                {
                    f"windspeed_{height}m": [8.5, 8.7, 8.3, 8.6, 8.4],
                    "timestamp": timestamps,
                    "year": timestamps.year,
                    "month": timestamps.month,
                    "hour": timestamps.hour,
                    "mohr": timestamps.month * 100
                    + timestamps.hour,  # month-hour combined
                }
            )

        mock_client.fetch_df = MagicMock(side_effect=mock_fetch_df)

        # Mock find_n_nearest_locations for grid-points endpoint
        def mock_find_n_nearest(lat, lon, limit=1):
            # Mock locations data - returns tuples of (index, lat, lon)
            locations = [
                ("046271", 39.903, -69.97427540107105),
                ("046272", 39.904, -69.98),
                ("046273", 39.905, -69.99),
                ("046274", 39.906, -70.00),
            ]
            return locations[:limit]

        mock_client.find_n_nearest_locations = MagicMock(
            side_effect=mock_find_n_nearest
        )

        # Set available heights (including 10m for legacy WTK endpoints)
        mock_client.available_heights = [
            10,
            30,
            40,
            50,
            60,
            80,
            100,
            120,
            140,
            160,
            200,
        ]
        mock_client.column_names = [
            "wind_speed_10m",
            "wind_speed_30m",
            "wind_speed_40m",
            "wind_speed_50m",
            "wind_speed_60m",
            "wind_speed_80m",
            "wind_speed_100m",
            "wind_speed_120m",
            "wind_speed_140m",
            "wind_speed_160m",
            "wind_speed_200m",
        ]

        return mock_client

    # Patch the windwatts_data client classes
    wtk_patch = patch(
        "windwatts_data.WindwattsWTKClient", return_value=create_mock_windwatts_client()
    )
    era5_patch = patch(
        "windwatts_data.WindwattsERA5Client",
        return_value=create_mock_windwatts_client(),
    )
    ensemble_patch = patch(
        "windwatts_data.WindwattsEnsembleClient",
        return_value=create_mock_windwatts_client(),
    )

    _windwatts_patches = [wtk_patch, era5_patch, ensemble_patch]
    for p in _windwatts_patches:
        p.start()

    # Still need boto3 mocks for other AWS services (like secrets manager in config_manager)
    mock_s3_client = MagicMock()
    mock_athena_client = MagicMock()

    _mock_clients = {
        "athena": mock_athena_client,
        "s3": mock_s3_client,
        "secretsmanager": MagicMock(),
    }

    def mock_boto3_client(service_name, *args, **kwargs):
        if service_name in _mock_clients:
            return _mock_clients[service_name]
        return MagicMock()

    _boto3_patch = patch("boto3.client", side_effect=mock_boto3_client)
    _boto3_patch.start()


def pytest_unconfigure(config):
    """
    Cleanup: stop all patches after tests complete.
    """
    global _boto3_patch, _windwatts_patches
    if _boto3_patch:
        _boto3_patch.stop()
    for p in _windwatts_patches:
        p.stop()
    try:
        patch.stopall()
    except Exception:
        pass
