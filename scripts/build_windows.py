#!/usr/bin/env python3
"""
HelixOne Windows Build Script
Automates the entire build process:
1. Convert PNG logo to ICO
2. Run PyInstaller
3. Compile Inno Setup installer
4. Generate checksums
5. Create version.json for updates
"""

import os
import sys
import json
import hashlib
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
DIST_DIR = PROJECT_ROOT / "dist"
INSTALLER_DIR = PROJECT_ROOT / "installer"
BUILD_DIR = PROJECT_ROOT / "build"

# Output paths
OUTPUT_DIR = DIST_DIR / "installer"

# Version info
DEFAULT_VERSION = "1.0.0"


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def print_step(step: int, text: str):
    """Print step info"""
    print(f"\n[{step}/5] {text}...")


def check_requirements():
    """Check if required tools are installed"""
    print_header("Checking Requirements")

    # Check Python
    print(f"  Python: {sys.version.split()[0]}")

    # Check PyInstaller
    try:
        import PyInstaller
        print(f"  PyInstaller: {PyInstaller.__version__}")
    except ImportError:
        print("  ERROR: PyInstaller not installed!")
        print("  Run: pip install pyinstaller")
        return False

    # Check Pillow (for icon conversion)
    try:
        from PIL import Image
        import PIL
        print(f"  Pillow: {PIL.__version__}")
    except ImportError:
        print("  WARNING: Pillow not installed (needed for icon conversion)")
        print("  Run: pip install pillow")

    # Check Inno Setup (Windows only)
    if sys.platform == 'win32':
        iscc_path = find_inno_setup()
        if iscc_path:
            print(f"  Inno Setup: {iscc_path}")
        else:
            print("  WARNING: Inno Setup not found")
            print("  Download from: https://jrsoftware.org/isinfo.php")

    return True


def find_inno_setup() -> str:
    """Find Inno Setup compiler path"""
    possible_paths = [
        r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"C:\Program Files\Inno Setup 6\ISCC.exe",
        r"C:\Program Files (x86)\Inno Setup 5\ISCC.exe",
        r"C:\Program Files\Inno Setup 5\ISCC.exe",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # Try to find in PATH
    result = shutil.which("ISCC")
    return result or ""


def convert_png_to_ico(png_path: Path, ico_path: Path, sizes: list = None):
    """Convert PNG to ICO with multiple sizes"""
    print_step(1, "Converting logo to Windows ICO format")

    if sizes is None:
        sizes = [16, 24, 32, 48, 64, 128, 256]

    try:
        from PIL import Image

        # Open PNG
        img = Image.open(png_path)

        # Convert to RGBA if necessary
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Create ICO with multiple sizes
        icons = []
        for size in sizes:
            resized = img.resize((size, size), Image.Resampling.LANCZOS)
            icons.append(resized)

        # Save as ICO
        icons[0].save(
            ico_path,
            format='ICO',
            sizes=[(s, s) for s in sizes],
            append_images=icons[1:]
        )

        print(f"  Created: {ico_path}")
        print(f"  Sizes: {sizes}")
        return True

    except ImportError:
        print("  ERROR: Pillow not installed")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def run_pyinstaller(spec_file: Path):
    """Run PyInstaller with spec file"""
    print_step(2, "Building executable with PyInstaller")

    try:
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--clean",
            "--noconfirm",
            str(spec_file)
        ]

        print(f"  Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("  ERROR: PyInstaller failed!")
            print(result.stderr)
            return False

        print("  Build successful!")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def run_inno_setup(iss_file: Path, version: str):
    """Compile installer with Inno Setup"""
    print_step(3, "Creating installer with Inno Setup")

    iscc_path = find_inno_setup()
    if not iscc_path:
        print("  ERROR: Inno Setup not found!")
        print("  Skipping installer creation...")
        return False

    try:
        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        cmd = [
            iscc_path,
            f"/DMyAppVersion={version}",
            str(iss_file)
        ]

        print(f"  Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("  ERROR: Inno Setup failed!")
            print(result.stderr)
            return False

        print("  Installer created successfully!")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def generate_checksums(installer_path: Path):
    """Generate SHA256 checksum for installer"""
    print_step(4, "Generating checksums")

    try:
        sha256_hash = hashlib.sha256()

        with open(installer_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)

        checksum = sha256_hash.hexdigest()

        # Save checksum to file
        checksum_file = installer_path.with_suffix(".sha256")
        with open(checksum_file, "w") as f:
            f.write(f"{checksum}  {installer_path.name}\n")

        print(f"  SHA256: {checksum}")
        print(f"  Saved to: {checksum_file}")

        return checksum

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def create_version_json(version: str, installer_path: Path, checksum: str):
    """Create version.json for auto-updater"""
    print_step(5, "Creating version.json for updates")

    try:
        # Get file size
        file_size = installer_path.stat().st_size if installer_path.exists() else 0

        version_info = {
            "version": version,
            "release_date": datetime.now().strftime("%Y-%m-%d"),
            "download_url": f"https://clever-conkies-89d13b.netlify.app/downloads/{installer_path.name}",
            "file_size": file_size,
            "checksum": checksum,
            "changelog": [
                "Nouvelle version disponible",
                "Corrections de bugs",
                "Ameliorations de performance"
            ],
            "mandatory": False,
            "min_version": "0.0.0"
        }

        # Save to multiple locations
        json_paths = [
            OUTPUT_DIR / "version.json",
            PROJECT_ROOT / "public" / "api" / "version.json",
        ]

        for json_path in json_paths:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(version_info, f, indent=2, ensure_ascii=False)
            print(f"  Created: {json_path}")

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    """Main build process"""
    parser = argparse.ArgumentParser(description="Build HelixOne for Windows")
    parser.add_argument(
        "--version", "-v",
        default=DEFAULT_VERSION,
        help=f"Version number (default: {DEFAULT_VERSION})"
    )
    parser.add_argument(
        "--skip-pyinstaller",
        action="store_true",
        help="Skip PyInstaller step (use existing build)"
    )
    parser.add_argument(
        "--skip-installer",
        action="store_true",
        help="Skip Inno Setup step"
    )
    args = parser.parse_args()

    print_header(f"HelixOne Windows Build - v{args.version}")

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Step 1: Convert logo
    png_path = ASSETS_DIR / "logo.png"
    ico_path = ASSETS_DIR / "logo.ico"

    if png_path.exists():
        if not ico_path.exists() or png_path.stat().st_mtime > ico_path.stat().st_mtime:
            convert_png_to_ico(png_path, ico_path)
        else:
            print_step(1, "Logo ICO already up to date")
    else:
        print(f"  WARNING: {png_path} not found")

    # Step 2: Run PyInstaller
    if not args.skip_pyinstaller:
        spec_file = PROJECT_ROOT / "HelixOne_Windows.spec"
        if not run_pyinstaller(spec_file):
            print("\nBuild failed at PyInstaller step")
            sys.exit(1)
    else:
        print_step(2, "Skipping PyInstaller (--skip-pyinstaller)")

    # Step 3: Run Inno Setup
    installer_name = f"HelixOne_Setup_{args.version}.exe"
    installer_path = OUTPUT_DIR / installer_name

    if not args.skip_installer and sys.platform == 'win32':
        iss_file = INSTALLER_DIR / "HelixOne.iss"
        if not run_inno_setup(iss_file, args.version):
            print("\nBuild failed at Inno Setup step")
            print("Note: Inno Setup only works on Windows")
    else:
        print_step(3, "Skipping Inno Setup (not on Windows or --skip-installer)")

    # Step 4: Generate checksums
    checksum = None
    if installer_path.exists():
        checksum = generate_checksums(installer_path)
    else:
        print_step(4, "Skipping checksums (installer not found)")

    # Step 5: Create version.json
    create_version_json(args.version, installer_path, checksum or "")

    # Summary
    print_header("Build Complete!")
    print(f"  Version: {args.version}")
    print(f"  Output: {OUTPUT_DIR}")

    if installer_path.exists():
        size_mb = installer_path.stat().st_size / (1024 * 1024)
        print(f"  Installer: {installer_name} ({size_mb:.1f} MB)")

    print("\nNext steps:")
    print("  1. Upload installer to Netlify /downloads/ folder")
    print("  2. Upload version.json to Netlify /api/ folder")
    print("  3. Test download from website")


if __name__ == "__main__":
    main()
