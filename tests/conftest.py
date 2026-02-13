"""
Pytest configuration and shared fixtures
"""

import sys
from pathlib import Path

import pytest

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def mock_session_file(tmp_path):
    """Create a temporary session file for testing"""
    session_file = tmp_path / ".helixone_session.json"
    return session_file


@pytest.fixture
def mock_quick_login_file(tmp_path):
    """Create a temporary quick login file for testing"""
    quick_login_file = tmp_path / ".helixone_quick_login.json"
    return quick_login_file


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing"""
    monkeypatch.setenv("HELIXONE_API_URL", "http://localhost:8000")
    monkeypatch.setenv("HELIXONE_TEST_MODE", "1")


@pytest.fixture
def sample_user_data():
    """Sample user data for testing"""
    return {
        "id": 1,
        "email": "test@example.com",
        "first_name": "Test",
        "last_name": "User",
        "created_at": "2024-01-01T00:00:00",
    }


@pytest.fixture
def sample_auth_response(sample_user_data):
    """Sample authentication response"""
    return {
        "access_token": "test_token_123",
        "token_type": "bearer",
        "user": sample_user_data,
    }
