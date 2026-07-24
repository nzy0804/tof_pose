"""Build MaixSense service modules as native Cython extensions."""

from __future__ import annotations

import os

from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup


PROTECTED_MODULES = {
    "ai_pb2": "ai_pb2.py",
    "ai_pb2_grpc": "ai_pb2_grpc.py",
    "scripts.grpc_server": "scripts/grpc_server.py",
    "tof_pose.input_image": "src/tof_pose/input_image.py",
    "tof_pose.realtime_service": "src/tof_pose/realtime_service.py",
    "tof_pose.tracking": "src/tof_pose/tracking.py",
    "tof_pose.person_distance": "src/tof_pose/person_distance.py",
    "tof_pose.pose_drawing": "src/tof_pose/pose_drawing.py",
    "tof_pose.object_storage": "src/tof_pose/object_storage.py",
    "tof_pose.model_bundle": "src/tof_pose/model_bundle.py",
    "tof_pose.paths": "src/tof_pose/paths.py",
    "tof_pose.scene_rate_controller": "src/tof_pose/scene_rate_controller.py",
}


Options.docstrings = False

extensions = [
    Extension(
        name,
        [source],
        define_macros=[("NDEBUG", "1")],
        extra_compile_args=["-O3", "-g0", "-fvisibility=hidden", "-fno-ident"],
    )
    for name, source in PROTECTED_MODULES.items()
]

setup(
    name="maixsense-protected-modules",
    ext_modules=cythonize(
        extensions,
        build_dir=os.path.join("build", "cython"),
        nthreads=max(1, min(8, os.cpu_count() or 1)),
        compiler_directives={
            "language_level": 3,
            "annotation_typing": False,
            "binding": False,
            "embedsignature": False,
            "linetrace": False,
            "profile": False,
            "emit_code_comments": False,
        },
    ),
)
