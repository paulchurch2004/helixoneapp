"""
Integration tests for API client
"""

import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.mark.skip(reason="Requires running backend server")
class TestAPIClient:
    """Integration tests for HelixOne API client"""

    def test_health_check(self):
        """Test API health check endpoint"""
        # TODO: Implement when backend is available
        pass

    def test_login_flow(self):
        """Test complete login flow"""
        # TODO: Implement integration test for login
        pass

    def test_get_user_profile(self):
        """Test retrieving user profile"""
        # TODO: Implement integration test
        pass
