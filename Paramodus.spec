# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

def get_collect_all(name):
    datas, binaries, hiddenimports = collect_all(name)
    return datas, binaries, hiddenimports

# Collect for heavy AI libraries
l_datas, l_binaries, l_hiddenimports = get_collect_all('lancedb')
f_datas, f_binaries, f_hiddenimports = get_collect_all('fastembed')
a_datas, a_binaries, a_hiddenimports = get_collect_all('agno')

# Main Analysis for Paramodus (based on app.py)
a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=l_binaries + f_binaries + a_binaries,
    datas=[
        # bin is excluded from the internal bundle to stay small
        # It will be provided by the installer in the same folder as the EXE
        ('ui', 'ui'),
    ] + l_datas + f_datas + a_datas,
    hiddenimports=[
        'pywebview',
        'onnxruntime',
    ] + l_hiddenimports + f_hiddenimports + a_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Paramodus',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False, # Set to True if you need to see terminal logs for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
