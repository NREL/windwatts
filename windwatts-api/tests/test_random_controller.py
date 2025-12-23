from fastapi.testclient import TestClient
from app.main import app
from app.utils.random_message import messages

client = TestClient(app)


def test_read_root():
    response = client.get("/random")
    json = response.json()
    assert response.status_code == 200
    assert "message" in json and json["message"] in messages


def test_read_chuck():
    response = client.get("/random/chuck")
    json = response.json()
    assert response.status_code == 200
    assert "joke" in json


def test_read_chuck_category():
    response = client.get("/random/chuck/dev")
    json = response.json()
    assert response.status_code == 200
    assert "joke" in json


def test_read_chuck_category_invalid():
    response = client.get("/random/chuck/invalid")
    json = response.json()
    assert response.status_code == 500
    assert "detail" in json
    assert "Category can only be one of:" in json["detail"]
