import pytest
from fastapi.testclient import TestClient
from api.main import app
from settings import settings

def test_auth_fail_closed_unconfigured():
    """Verify that when no token is provided, the API rejects the request."""
    # Temporarily remove dependency overrides set by conftest.py's client fixture
    # to test the REAL dependency.
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides.clear()
    
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 401
        assert "API Key unconfigured" in response.json()["detail"] or "Invalid or missing" in response.json()["detail"]
        
    app.dependency_overrides = original_overrides

def test_auth_fail_closed_wrong_token(monkeypatch):
    """Verify that a wrong token is rejected when API_KEY is configured."""
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides.clear()
    
    monkeypatch.setattr(settings, "API_KEY", "real-test-token")
    
    with TestClient(app) as client:
        response = client.get("/health", headers={"Authorization": "Bearer wrong-token"})
        assert response.status_code == 401
        assert "Invalid or missing" in response.json()["detail"]
        
    app.dependency_overrides = original_overrides

def test_auth_success_correct_token(monkeypatch):
    """Verify that the correct token succeeds."""
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides.clear()
    
    # We must mock settings.API_KEY because it's checked by verify_token
    monkeypatch.setattr(settings, "API_KEY", "real-test-token")
    
    with TestClient(app) as client:
        response = client.get("/health", headers={"Authorization": "Bearer real-test-token"})
        assert response.status_code == 200
        assert response.json()["data"]["status"] == "ok"
        
    app.dependency_overrides = original_overrides
