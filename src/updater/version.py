"""
HelixOne Version Management
"""

# Current application version
CURRENT_VERSION = "1.0.5"

# Build info
BUILD_DATE = "2026-02-12"
BUILD_NUMBER = 1

def get_version_info() -> dict:
    """Get complete version information"""
    return {
        "version": CURRENT_VERSION,
        "build_date": BUILD_DATE,
        "build_number": BUILD_NUMBER,
        "full_version": f"{CURRENT_VERSION}.{BUILD_NUMBER}",
    }

def parse_version(version_string: str) -> tuple:
    """Parse version string to tuple for comparison"""
    try:
        parts = version_string.split('.')
        return tuple(int(p) for p in parts)
    except (ValueError, AttributeError):
        return (0, 0, 0)

def compare_versions(v1: str, v2: str) -> int:
    """
    Compare two version strings.
    Returns:
        -1 if v1 < v2
         0 if v1 == v2
         1 if v1 > v2
    """
    v1_tuple = parse_version(v1)
    v2_tuple = parse_version(v2)

    if v1_tuple < v2_tuple:
        return -1
    elif v1_tuple > v2_tuple:
        return 1
    else:
        return 0

def is_update_available(remote_version: str) -> bool:
    """Check if remote version is newer than current"""
    return compare_versions(remote_version, CURRENT_VERSION) > 0
