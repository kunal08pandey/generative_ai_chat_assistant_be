def test_logs_require_admin_key(client):
    unauthorized = client.get("/logs/")
    assert unauthorized.status_code == 401

    authorized = client.get("/logs/", headers={"X-Admin-Key": "test-admin-key"})
    assert authorized.status_code == 200
    assert "lines" in authorized.json()["data"]
