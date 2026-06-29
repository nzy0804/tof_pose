# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the TensorRT-enabled gRPC server binary.

This package is intended to run with explicit *.engine model paths, so model
assets are excluded by default to keep the binary smaller. Set
MAIXSENSE_INCLUDE_ASSETS=1 when building if the default *.pt assets are needed.
"""

import os

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules


def _safe_collect_submodules(package: str) -> list[str]:
    try:
        return collect_submodules(package)
    except Exception:
        return []


def _safe_collect_dynamic_libs(package: str) -> list[tuple[str, str]]:
    try:
        return collect_dynamic_libs(package)
    except Exception:
        return []


def _safe_collect_data_files(package: str) -> list[tuple[str, str]]:
    try:
        return collect_data_files(package, includes=["**/*.so*", "**/*.json"])
    except Exception:
        return []


hiddenimports = [
    "tof_pose.realtime_service",
    "tof_pose.object_storage",
    "tof_pose.paths",
    "tof_pose.person_distance",
    "tof_pose.pose_drawing",
    "ultralytics",
    "ultralytics.nn.backends.tensorrt",
    "cv2",
    "grpc",
    "numpy",
    # Ultralytics trackers (BoT-SORT/ByteTrack) linear assignment dependency.
    "lap",
    "lap._lapjv",
    # TensorRT runtime packages used by Ultralytics when loading *.engine files.
    "tensorrt",
    "tensorrt_bindings",
    "tensorrt_libs",
    "cuda",
    "cuda.bindings",
]

for package in (
    "tof_pose",
    "lap",
    "oss2",
    "boto3",
    "botocore",
    "s3transfer",
    "tensorrt",
    "tensorrt_bindings",
    "tensorrt_libs",
    "cuda",
    "cuda.bindings",
):
    hiddenimports += _safe_collect_submodules(package)

binaries = []
datas = []
tof_pose_assets_dir = os.path.join("src", "tof_pose", "assets")
if os.path.isdir(tof_pose_assets_dir):
    datas.append((tof_pose_assets_dir, os.path.join("tof_pose", "assets")))

for package in (
    "oss2",
    "boto3",
    "botocore",
    "s3transfer",
    "tensorrt",
    "tensorrt_bindings",
    "tensorrt_libs",
    "cuda",
    "cuda.bindings",
):
    binaries += _safe_collect_dynamic_libs(package)
    datas += _safe_collect_data_files(package)

if os.environ.get("MAIXSENSE_INCLUDE_ASSETS") == "1" and os.path.isdir("assets"):
    datas.append(("assets", "assets"))


a = Analysis(
    [os.path.join("scripts", "grpc_server.py")],
    pathex=[os.path.abspath("."), os.path.abspath("src")],
    binaries=binaries,
    datas=datas,
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
    [],
    name="maixsense-grpc-server",
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
    exclude_binaries=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="maixsense-grpc-server",
)
