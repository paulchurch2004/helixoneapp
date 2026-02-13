"""
Unit tests for version module
"""

import pytest
from updater.version import (
    CURRENT_VERSION,
    compare_versions,
    get_version_info,
    is_update_available,
    parse_version,
)


class TestVersion:
    """Test suite for version utilities"""

    def test_current_version_format(self):
        """Test that CURRENT_VERSION is in correct format"""
        assert isinstance(CURRENT_VERSION, str)
        parts = CURRENT_VERSION.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)

    def test_get_version_info(self):
        """Test getting version information"""
        info = get_version_info()
        assert "version" in info
        assert "build_date" in info
        assert "build_number" in info
        assert "full_version" in info
        assert info["version"] == CURRENT_VERSION

    @pytest.mark.parametrize(
        "version_str,expected",
        [
            ("1.0.0", (1, 0, 0)),
            ("2.5.10", (2, 5, 10)),
            ("0.0.1", (0, 0, 1)),
            ("invalid", (0, 0, 0)),
        ],
    )
    def test_parse_version(self, version_str, expected):
        """Test version string parsing"""
        result = parse_version(version_str)
        assert result == expected

    @pytest.mark.parametrize(
        "version1,version2,expected",
        [
            ("1.0.0", "1.0.1", -1),
            ("1.0.1", "1.0.0", 1),
            ("1.0.0", "1.0.0", 0),
            ("1.0.5", "1.1.0", -1),
            ("2.0.0", "1.9.9", 1),
        ],
    )
    def test_compare_versions(self, version1, version2, expected):
        """Test version comparison"""
        result = compare_versions(version1, version2)
        assert result == expected

    @pytest.mark.parametrize(
        "remote,expected",
        [
            ("1.0.6", True),  # Newer patch version
            ("1.1.0", True),  # Newer minor version
            ("2.0.0", True),  # Newer major version
            ("1.0.5", False),  # Same version
            ("1.0.4", False),  # Older version
        ],
    )
    def test_is_update_available(self, remote, expected):
        """Test update availability check"""
        result = is_update_available(remote)
        assert result == expected
