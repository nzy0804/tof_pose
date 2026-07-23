# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec that packages only compiled MaixSense service modules."""

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


def _is_inference_runtime_file(item: tuple[str, ...]) -> bool:
    return all(
        "libnvinfer_builder_resource" not in os.path.basename(path)
        for path in item[:2]
    )


protected_modules = [
    "ai_pb2",
    "ai_pb2_grpc",
    "scripts.grpc_server",
    "tof_pose.realtime_service",
    "tof_pose.tracking",
    "tof_pose.object_storage",
    "tof_pose.model_bundle",
    "tof_pose.paths",
    "tof_pose.person_distance",
    "tof_pose.pose_drawing",
]

hiddenimports = protected_modules + [
    "ultralytics",
    "ultralytics.nn.backends.tensorrt",
    "cv2",
    "grpc",
    "numpy",
    "lap",
    "lap._lapjv",
    "tensorrt",
    "tensorrt_bindings",
    "tensorrt_libs",
    "cuda",
    "cuda.bindings",
    "cryptography",
]

for package in (
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
    "cryptography",
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
    "cryptography",
):
    binaries += [
        item for item in _safe_collect_dynamic_libs(package)
        if _is_inference_runtime_file(item)
    ]
    datas += [
        item for item in _safe_collect_data_files(package)
        if _is_inference_runtime_file(item)
    ]


a = Analysis(
    [os.path.join("scripts", "grpc_server_bootstrap.py")],
    pathex=[os.path.abspath("."), os.path.abspath("src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=2,
)

# Built-in hooks may add builder resources after the explicit collection above.
a.binaries = [item for item in a.binaries if _is_inference_runtime_file(item)]
a.datas = [item for item in a.datas if _is_inference_runtime_file(item)]

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
