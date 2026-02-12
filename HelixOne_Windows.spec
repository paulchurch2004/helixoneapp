# -*- mode: python ; coding: utf-8 -*-
"""
HelixOne - PyInstaller Spec for Windows
Generates a Windows executable (.exe) for distribution
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

PROJECT_ROOT = os.path.dirname(os.path.abspath(SPEC))

datas = [
    (os.path.join(PROJECT_ROOT, 'src'), 'src'),
    (os.path.join(PROJECT_ROOT, 'assets'), 'assets'),
    (os.path.join(PROJECT_ROOT, 'data', 'formation_commerciale'),
     os.path.join('data', 'formation_commerciale')),
    (os.path.join(PROJECT_ROOT, 'src', 'i18n'), os.path.join('src', 'i18n')),
]

datas += collect_data_files('customtkinter')

hiddenimports = [
    'customtkinter',
    'customtkinter.windows',
    'customtkinter.windows.widgets',
    'PIL', 'PIL.Image', 'PIL.ImageTk',
    'pandas', 'numpy',
    'requests', 'urllib3',
    'yfinance',
    'cv2', 'pygame',
    'src', 'src.interface', 'src.interface.main_app',
    'src.interface.main_window', 'src.interface.login_window',
    'src.interface.home_panel', 'src.interface.formation_commerciale',
    'src.plasma_intro', 'src.auth_session', 'src.auth_manager',
    'src.helixone_client', 'src.config', 'src.i18n', 'src.asset_path',
    'src.updater', 'src.updater.auto_updater', 'src.updater.version',
    'src.secure_storage', 'src.biometric_auth', 'src.device_manager',
    'keyring', 'keyring.backends',
]

hiddenimports += collect_submodules('customtkinter')
hiddenimports += collect_submodules('src.interface')

excludes = [
    'tkinter.test', 'unittest', 'pytest',
    'IPython', 'jupyter', 'notebook', 'sphinx', 'docutils',
]

a = Analysis(
    ['run_packaged.py'],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HelixOne',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(PROJECT_ROOT, 'assets', 'logo.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HelixOne',
)
