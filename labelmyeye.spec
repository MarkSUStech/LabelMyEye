# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for LabelMyEye."""

block_cipher = None

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets/models/mobile_sam_encoder.onnx', 'assets/models'),
        ('assets/models/mobile_sam_encoder.onnx.data', 'assets/models'),
        ('assets/models/mobile_sam_decoder.onnx', 'assets/models'),
        ('assets/models/mobile_sam_decoder.onnx.data', 'assets/models'),
    ],
    hiddenimports=[
        'labelmyeye',
        'labelmyeye.app',
        'labelmyeye.canvas',
        'labelmyeye.shape',
        'labelmyeye.label_file',
        'labelmyeye.history',
        'labelmyeye.dr_geometry',
        'labelmyeye.sam_seg',
        'labelmyeye.auto_annotate',
        'labelmyeye.model_setup',
        'labelmyeye.file_panel',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LabelMyEye',
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
    icon='assets/labelmyeye.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LabelMyEye',
)
