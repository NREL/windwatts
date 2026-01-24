"""
Tests for V1 API endpoints.

"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestV1WindspeedEndpoints:
    """Test windspeed endpoints for all models."""

    def test_era5_windspeed_default(self):
        """Test ERA5 windspeed with default period."""
        response = client.get(
            "/api/v1/era5-quantiles/windspeed?lat=40.0&lng=-70.0&height=40"
        )
        assert response.status_code == 200
        json = response.json()
        assert "global_avg" in json

    def test_era5_windspeed_all(self):
        """Test ERA5 windspeed with period=all."""
        response = client.get(
            "/api/v1/era5-quantiles/windspeed?lat=40.0&lng=-70.0&height=40&period=all"
        )
        assert response.status_code == 200
        json = response.json()
        assert "global_avg" in json

    def test_era5_windspeed_annual(self):
        """Test ERA5 windspeed with period=annual."""
        response = client.get(
            "/api/v1/era5-quantiles/windspeed?lat=40.0&lng=-70.0&height=40&period=annual"
        )
        assert response.status_code == 200
        json = response.json()
        assert isinstance(json, dict)
        # Should have yearly_avg key with array of year data
        assert "yearly_avg" in json
        assert isinstance(json["yearly_avg"], list)
        if json["yearly_avg"]:
            assert "year" in json["yearly_avg"][0]

    def test_wtk_windspeed_default(self):
        """Test WTK windspeed with default period."""
        response = client.get(
            "/api/v1/wtk-timeseries/windspeed?lat=40.0&lng=-100.0&height=80"
        )
        assert response.status_code == 200
        json = response.json()
        assert "global_avg" in json

    def test_ensemble_windspeed(self):
        """Test ensemble windspeed."""
        response = client.get(
            "/api/v1/ensemble-quantiles/windspeed?lat=40.0&lng=-70.0&height=40"
        )
        assert response.status_code == 200
        json = response.json()
        assert "global_avg" in json

    def test_windspeed_invalid_model(self):
        """Test windspeed with invalid model."""
        response = client.get(
            "/api/v1/invalid_model/windspeed?lat=40.0&lng=-70.0&height=40"
        )
        assert response.status_code == 400
        json = response.json()
        assert "detail" in json
        assert "Invalid model" in json["detail"]

    def test_windspeed_invalid_period(self):
        """Test windspeed with invalid period."""
        response = client.get(
            "/api/v1/era5-quantiles/windspeed?lat=40.0&lng=-70.0&height=40&period=invalid"
        )
        assert response.status_code == 400


class TestV1ProductionEndpoints:
    """Test production endpoints for all models."""

    def test_era5_production_all(self):
        """Test ERA5 production with period=all."""
        response = client.get(
            "/api/v1/era5-quantiles/production?lat=40.0&lng=-70.0&height=40&powercurve=nlr-reference-100kW&period=all"
        )
        assert response.status_code == 200
        json = response.json()
        assert "energy_production" in json
        assert isinstance(json["energy_production"], (int, float))

    def test_era5_production_summary(self):
        """Test ERA5 production with period=summary."""
        response = client.get(
            "/api/v1/era5-quantiles/production?lat=40.0&lng=-70.0&height=40&powercurve=nlr-reference-100kW&period=summary"
        )
        assert response.status_code == 200
        json = response.json()
        assert "summary_avg_energy_production" in json

    def test_era5_production_annual(self):
        """Test ERA5 production with period=annual."""
        response = client.get(
            "/api/v1/era5-quantiles/production?lat=40.0&lng=-70.0&height=40&powercurve=nlr-reference-100kW&period=annual"
        )
        assert response.status_code == 200
        json = response.json()
        assert "yearly_avg_energy_production" in json

    def test_era5_production_full(self):
        """Test ERA5 production with period=full."""
        response = client.get(
            "/api/v1/era5-quantiles/production?lat=40.0&lng=-70.0&height=40&powercurve=nlr-reference-100kW&period=full"
        )
        assert response.status_code == 200
        json = response.json()
        assert "energy_production" in json
        assert "summary_avg_energy_production" in json
        assert "yearly_avg_energy_production" in json

    def test_wtk_production(self):
        """Test WTK production."""
        response = client.get(
            "/api/v1/wtk-timeseries/production?lat=40.0&lng=-100.0&height=80&powercurve=nlr-reference-100kW&period=all"
        )
        assert response.status_code == 200
        json = response.json()
        assert "energy_production" in json

    def test_ensemble_production(self):
        """Test ensemble production (only supports period=all)."""
        response = client.get(
            "/api/v1/ensemble-quantiles/production?lat=40.0&lng=-70.0&height=40&powercurve=nlr-reference-100kW&period=all"
        )
        assert response.status_code == 200
        json = response.json()
        assert "energy_production" in json

    def test_ensemble_production_full_fails(self):
        """Test ensemble production with period=full should fail."""
        response = client.get(
            "/api/v1/ensemble-quantiles/production?lat=40.0&lng=-70.0&height=40&powercurve=nlr-reference-100kW&period=full"
        )
        assert response.status_code == 400

    def test_production_invalid_powercurve(self):
        """Test production with invalid power curve."""
        response = client.get(
            "/api/v1/era5-quantiles/production?lat=40.0&lng=-70.0&height=40&powercurve=invalid-curve&period=all"
        )
        assert response.status_code == 400


class TestV1PowerCurves:
    """Test power curves endpoint."""

    def test_get_powercurves(self):
        """Test getting available power curves."""
        response = client.get("/api/v1/powercurves")
        assert response.status_code == 200
        json = response.json()
        assert "available_power_curves" in json
        assert isinstance(json["available_power_curves"], list)
        assert len(json["available_power_curves"]) > 0
        # Check that nlr-reference curves are present
        assert any("nlr-reference" in pc for pc in json["available_power_curves"])


class TestV1GridPoints:
    """Test grid points endpoints."""

    def test_era5_grid_points(self):
        """Test ERA5 grid points lookup."""
        response = client.get(
            "/api/v1/era5-quantiles/grid-points?lat=40.0&lng=-70.0&limit=1"
        )
        assert response.status_code == 200
        json = response.json()
        assert "locations" in json
        assert isinstance(json["locations"], list)
        assert len(json["locations"]) == 1
        assert "index" in json["locations"][0]
        assert "latitude" in json["locations"][0]
        assert "longitude" in json["locations"][0]

    def test_grid_points_multiple_neighbors(self):
        """Test grid points with multiple neighbors."""
        response = client.get(
            "/api/v1/era5-quantiles/grid-points?lat=40.0&lng=-70.0&limit=4"
        )
        assert response.status_code == 200
        json = response.json()
        assert len(json["locations"]) == 4

    def test_wtk_grid_points(self):
        """Test WTK grid points lookup."""
        response = client.get(
            "/api/v1/wtk-timeseries/grid-points?lat=40.0&lng=-100.0&limit=1"
        )
        assert response.status_code == 200
        json = response.json()
        assert "locations" in json


class TestV1Validation:
    """Test parameter validation."""

    def test_invalid_latitude(self):
        """Test with invalid latitude."""
        response = client.get(
            "/api/v1/era5-quantiles/windspeed?lat=100.0&lng=-70.0&height=40"
        )
        assert response.status_code == 400

    def test_invalid_longitude(self):
        """Test with invalid longitude."""
        response = client.get(
            "/api/v1/era5-quantiles/windspeed?lat=40.0&lng=-200.0&height=40"
        )
        assert response.status_code == 400

    def test_invalid_height(self):
        """Test with invalid height."""
        response = client.get(
            "/api/v1/era5-quantiles/windspeed?lat=40.0&lng=-70.0&height=500"
        )
        assert response.status_code == 400

    def test_missing_required_params(self):
        """Test with missing required parameters."""
        response = client.get("/api/v1/era5-quantiles/windspeed")
        assert response.status_code == 422  # FastAPI validation error


class TestV1Info:
    """Test info endpoint."""

    def test_era5_info(self):
        """Test ERA5 info endpoint."""
        response = client.get("/api/v1/era5-quantiles/info")
        assert response.status_code == 200
        json = response.json()
        assert "model" in json
        assert json["model"] == "era5-quantiles"
        assert "supported_periods" in json
        assert "available_heights" in json

    def test_wtk_info(self):
        """Test WTK info endpoint."""
        response = client.get("/api/v1/wtk-timeseries/info")
        assert response.status_code == 200
        json = response.json()
        assert json["model"] == "wtk-timeseries"

    def test_ensemble_info(self):
        """Test ensemble info endpoint."""
        response = client.get("/api/v1/ensemble-quantiles/info")
        assert response.status_code == 200
        json = response.json()
        assert json["model"] == "ensemble-quantiles"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
