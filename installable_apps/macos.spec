# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(10000)

a = Analysis(
    ['desktop_application/startup.py'],
    pathex=[],
    binaries=[
        ('venv/lib/python3.12/site-packages/pgserver/pginstall', 'pgserver/pginstall')
    ],
    datas=[
        ('venv/lib/python3.12/site-packages/letta/server/static_files', 'letta/server/static_files'),
        ('venv/lib/python3.12/site-packages/letta/humans/examples', 'letta/humans/examples'),
        ('venv/lib/python3.12/site-packages/letta/personas/examples', 'letta/personas/examples'),
        ('venv/lib/python3.12/site-packages/letta/prompts/system', 'letta/prompts/system'),
        ('venv/lib/python3.12/site-packages/letta/functions/function_sets', 'letta/functions/function_sets'),
        ('assets','assets'),
        ('desktop_application/templates','templates'),

    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='letta',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets/letta.icns'],
)
app = BUNDLE(
    exe,
    name='letta.app',
    icon='assets/letta.icns',
)
