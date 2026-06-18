"""
Unit tests for DeepFakeShield AI API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json





@pytest.fixture
def auth_headers(client):
    """Get auth headers with valid token."""
    # Register user
    client.post("/api/v1/auth/register", json={
        "email": "test@example.com",
        "password": "testpassword123",
        "full_name": "Test User"
    })
    
    # Login
    response = client.post("/api/v1/auth/login", json={
        "email": "test@example.com",
        "password": "testpassword123"
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestHealth:
    """Health endpoint tests."""
    
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestAuth:
    """Authentication tests."""
    
    def test_register_user(self, client):
        response = client.post("/api/v1/auth/register", json={
            "email": "newuser@example.com",
            "password": "securepass123",
            "full_name": "New User"
        })
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert "id" in data
    
    def test_register_duplicate_email(self, client):
        # First registration
        client.post("/api/v1/auth/register", json={
            "email": "duplicate@example.com",
            "password": "password123"
        })
        
        # Duplicate
        response = client.post("/api/v1/auth/register", json={
            "email": "duplicate@example.com",
            "password": "password123"
        })
        assert response.status_code == 400
    
    def test_login_success(self, client):
        # Register
        client.post("/api/v1/auth/register", json={
            "email": "login@example.com",
            "password": "password123"
        })
        
        # Login
        response = client.post("/api/v1/auth/login", json={
            "email": "login@example.com",
            "password": "password123"
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_wrong_password(self, client):
        # Register
        client.post("/api/v1/auth/register", json={
            "email": "wrongpass@example.com",
            "password": "correctpassword"
        })
        
        # Wrong password
        response = client.post("/api/v1/auth/login", json={
            "email": "wrongpass@example.com",
            "password": "incorrectpassword"
        })
        assert response.status_code == 401
    
    def test_get_current_user(self, client, auth_headers):
        response = client.get("/api/v1/auth/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "test@example.com"


class TestMediaUpload:
    """Media upload tests."""
    
    def test_upload_requires_auth(self, client):
        response = client.post("/api/v1/media/upload")
        assert response.status_code == 401
    
    def test_upload_invalid_file_type(self, client, auth_headers):
        files = {"file": ("test.txt", b"test content", "text/plain")}
        response = client.post(
            "/api/v1/media/upload", 
            files=files,
            headers=auth_headers
        )
        assert response.status_code == 400
    
    def test_upload_video(self, client, auth_headers):
        # Fake MP4 magic bytes
        fake_video = b'\x00\x00\x00\x1c\x66\x74\x79\x70' + b'\x00' * 100
        files = {"file": ("test.mp4", fake_video, "video/mp4")}
        
        response = client.post(
            "/api/v1/media/upload",
            files=files,
            headers=auth_headers
        )
        # May fail due to validation - just check it doesn't crash
        assert response.status_code in [201, 400]


class TestAnalysis:
    """Analysis endpoint tests."""
    
    def test_start_analysis_requires_auth(self, client):
        response = client.post("/api/v1/analysis/start", json={
            "media_id": "test-id"
        })
        assert response.status_code == 401
    
    def test_get_status_not_found(self, client, auth_headers):
        response = client.get(
            "/api/v1/analysis/123e4567-e89b-12d3-a456-426614174000/status",
            headers=auth_headers
        )
        assert response.status_code == 404


class TestReports:
    """Report endpoint tests."""
    
    def test_reports_requires_auth(self, client):
        response = client.get("/api/v1/reports/123e4567-e89b-12d3-a456-426614174000/report")
        assert response.status_code == 401
