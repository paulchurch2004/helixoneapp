# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['run_packaged.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('src', 'src'),
        ('assets', 'assets'),
        ('data/formation_commerciale', 'data/formation_commerciale'),
        ('src/i18n', 'src/i18n'),
    ],
    hiddenimports=[
        'customtkinter',
        'PIL', 'PIL.Image', 'PIL.ImageTk',
        'requests', 'urllib3',
        'pandas', 'numpy',
        'yfinance',
        'src', 'src.interface', 'src.interface.main_app',
        'src.interface.main_window', 'src.interface.login_window',
        'src.interface.home_panel', 'src.interface.formation_commerciale',
        'src.interface.settings_panel', 'src.interface.profile_panel',
        'src.interface.boot_intro', 'src.interface.matrix_engine',
        'src.interface.theme_manager', 'src.interface.design_system',
        'src.interface.ui_components', 'src.interface.toast_notification',
        'src.interface.keyboard_shortcuts', 'src.interface.license_gatekeeper',
        'src.interface.register_window', 'src.interface.tooltips',
        'src.plasma_intro', 'src.auth_session', 'src.auth_manager',
        'src.helixone_client', 'src.config', 'src.i18n', 'src.asset_path',
        'src.updater', 'src.updater.auto_updater', 'src.updater.version',
        'src.security', 'src.security.license_manager', 'src.security.secure_config',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter.test', 'unittest', 'pytest',
        'IPython', 'jupyter', 'notebook', 'sphinx',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

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
    icon=['assets/logo.icns'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HelixOne',
)
app = BUNDLE(
    coll,
    name='HelixOne.app',
    icon='assets/logo.icns',
    bundle_identifier='com.helixone.app',
)
