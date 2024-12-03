# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['startup.py'],
    pathex=[],
    binaries=[
        ('venv/lib/python3.12/site-packages/pgserver/pginstall', 'pgserver/pginstall')
    ],
    datas=[
        ('venv/lib/python3.12/site-packages/letta/humans/examples', 'letta/humans/examples'),
        ('venv/lib/python3.12/site-packages/letta/personas/examples', 'letta/personas/examples'),
        ('venv/lib/python3.12/site-packages/letta/prompts/system', 'letta/prompts/system'),
        ('venv/lib/python3.12/site-packages/letta/functions/function_sets', 'letta/functions/function_sets')
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
    [],
    exclude_binaries=True,
    name='letta',
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
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='letta',
)
