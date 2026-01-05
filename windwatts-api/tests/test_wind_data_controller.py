from fastapi.testclient import TestClient
from app.main import app
# from unittest.mock import patch

client = TestClient(app)

# uncomment these when i can get local athena_config working
"""
# this patches the fetch_data method on the data_fetcher_router instance
def test_get_wtk_data_success():
    # Fake data to be returned by the mocked fetch_data call.
    fake_data = {"global_avg": 5.5}
    # Patch the fetch_data method on the data_fetcher_router instance.
    with patch("app.controllers.wind_data_controller.data_fetcher_router.fetch_data", return_value=fake_data):
        response = client.get("/wtk-data?lat=40.0&lng=-70.0&height=10&source=athena")
        assert response.status_code == 200
        assert response.json() == fake_data

# this patches the fetch_data method on the data_fetcher_router instance
def test_get_wtk_data_failure():
    # Patch the fetch_data method on the data_fetcher_router instance to raise an exception.
    with patch("app.controllers.wind_data_controller.data_fetcher_router.fetch_data", side_effect=Exception("Test exception")):
        response = client.get("/wtk-data?lat=40.0&lng=-70.0&height=10&source=athena")
        assert response.status_code == 500
        assert response.json() == {"detail": "Test exception"}
"""


def test_get_wtk_data_success():
    response = client.get(
        "/wtk/windspeed?lat=40.0&lng=-70.0&height=10&source=athena_wtk"
    )
    assert response.status_code == 200
    json = response.json()
    assert "global_avg" in json


def test_get_available_power_curves():
    response = client.get("/wtk/available-powercurves")
    assert response.status_code == 200
    json = response.json()
    assert "available_power_curves" in json
