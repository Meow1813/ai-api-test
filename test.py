import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_valid_url(client):
    response = client.get(f"/get-photo-class",
                          params={
                              'url': 'https://sun9-5.userapi.com/impg/rCR57pxqt7zUWiMyznlmhJ_EYmDHPvow_5aVRQ/L0TCWyTRR44.jpg?size=900x900&quality=95&sign=5765bc79982ec568aedc5f78c998f062&type=album',
                          })
    assert response.status_code == 200
    assert "Predicted class:" in response.json()


def test_get_photo_class_invalid_url(client):
    response = client.get(f"/get-photo-class",
                          params={
                              'url': 'invalid_url',
                          })
    assert response.status_code == 422
    assert "Invalid URL" in response.text
