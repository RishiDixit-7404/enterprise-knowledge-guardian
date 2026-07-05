def test_health_check(client):
    """Test that GET /health returns 200 HTTP status and conforms to the response envelope."""
    response = client.get("/health")
    assert response.status_code == 200
    
    body = response.json()
    assert body["success"] is True
    assert body["status_code"] == 200
    assert body["error_message"] == ""
    assert body["data"] == {"status": "ok"}
