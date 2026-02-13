"""
Unit tests for AuthManager
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from auth_manager import AuthManager


class TestAuthManager:
    """Test suite for AuthManager class"""

    @pytest.fixture
    def auth_manager(self, tmp_path):
        """Create an AuthManager instance for testing"""
        session_file = tmp_path / ".helixone_session.json"
        with patch("auth_manager.SESSION_FILE", str(session_file)):
            manager = AuthManager()
            yield manager

    def test_init(self, auth_manager):
        """Test AuthManager initialization"""
        assert auth_manager.client is not None
        assert auth_manager.get_current_user() is None

    @patch("auth_manager.HelixOneClient")
    def test_login_success(self, mock_client_class, auth_manager, sample_auth_response):
        """Test successful login"""
        mock_client = MagicMock()
        mock_client.login.return_value = sample_auth_response
        mock_client.user = sample_auth_response["user"]
        auth_manager.client = mock_client

        result = auth_manager.login("test@example.com", "password123")

        assert result == sample_auth_response
        assert auth_manager.get_current_user() == sample_auth_response["user"]
        mock_client.login.assert_called_once_with("test@example.com", "password123")

    @patch("auth_manager.HelixOneClient")
    def test_login_failure(self, mock_client_class, auth_manager):
        """Test failed login"""
        mock_client = MagicMock()
        mock_client.login.side_effect = Exception("Invalid credentials")
        auth_manager.client = mock_client

        with pytest.raises(Exception) as exc_info:
            auth_manager.login("test@example.com", "wrong_password")

        assert "Invalid credentials" in str(exc_info.value)

    def test_logout(self, auth_manager, sample_user_data, tmp_path):
        """Test logout functionality"""
        # Setup: simulate logged in state
        session_file = tmp_path / ".helixone_session.json"
        with patch("auth_manager.SESSION_FILE", str(session_file)):
            session_data = {"user": sample_user_data, "token": "test_token"}
            session_file.write_text(json.dumps(session_data))
            auth_manager.client.user = sample_user_data

            # Perform logout
            auth_manager.logout()

            # Verify
            assert auth_manager.get_current_user() is None
            assert not session_file.exists()

    def test_save_and_load_session(self, auth_manager, sample_user_data, tmp_path):
        """Test session persistence"""
        session_file = tmp_path / ".helixone_session.json"
        with patch("auth_manager.SESSION_FILE", str(session_file)):
            # Save session
            auth_manager.client.user = sample_user_data
            auth_manager.save_session("test_token_123")

            # Verify file was created
            assert session_file.exists()

            # Load session in new instance
            new_manager = AuthManager()
            loaded_user = new_manager.get_current_user()

            assert loaded_user is not None
            assert loaded_user["email"] == "test@example.com"

    @pytest.mark.unit
    def test_get_current_user_when_not_logged_in(self, auth_manager):
        """Test getting current user when not logged in"""
        user = auth_manager.get_current_user()
        assert user is None

    @pytest.mark.unit
    def test_is_authenticated(self, auth_manager, sample_user_data):
        """Test authentication status check"""
        # Not authenticated initially
        assert not auth_manager.is_authenticated()

        # Set user data
        auth_manager.client.user = sample_user_data

        # Now authenticated
        assert auth_manager.is_authenticated()
