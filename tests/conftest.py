import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture(scope="module")
def client():
    """Provides a TestClient for testing the FastAPI application endpoints."""
    with TestClient(app) as c:
        yield c
