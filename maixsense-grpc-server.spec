# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = [
    'tof_pose.realtime_service',
    'tof_pose.object_storage',
    'tof_pose.paths',
    'tof_pose.person_distance',
    'tof_pose.pose_drawing',
    'ultralytics',
    'cv2',
    'grpc',
    'numpy',
    # Ultralytics trackers (BoT-SORT/ByteTrack) linear assignment dependency.
    'lap',
    'lap._lapjv',
]
hiddenimports += collect_submodules('tof_pose')

try:
    hiddenimports += collect_submodules('lap')
except Exception:
    pass

for package in ('oss2', 'boto3', 'botocore', 's3transfer'):
    try:
        hiddenimports += collect_submodules(package)
    except Exception:
        pass


a = Analysis(
    [os.path.join('scripts', 'grpc_server.py')],
    pathex=[os.path.abspath('.'), os.path.abspath('src')],
    binaries=[],
    datas=[('assets', 'assets')],
    hiddenimports=hiddenimports,
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
    name='maixsense-grpc-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
