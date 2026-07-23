"""Load an authenticated multi-component model package."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


_MAGIC = b"MSB1"
_VERSION = 1
_COMPONENT_COUNT = 2
_HEADER = struct.Struct(">4sBBH")
_ENTRY = struct.Struct(">12sQ32s")
_SEGMENT_NAMES = {
    str(index): "person" if index == 0 else f"reserved_{index}"
    for index in range(80)
}
_ENGINE_METADATA = (
    {"task": "segment", "names": _SEGMENT_NAMES},
    {"task": "pose", "names": {"0": "person"}, "kpt_shape": [17, 3]},
)


class ModelBundleMaterializer:
    """Decrypt model components into a private temporary directory."""

    def __init__(self, bundle_path: str | Path, key_path: str | Path) -> None:
        self._bundle_path = Path(bundle_path)
        self._key_path = Path(key_path)
        self._temporary_dir: Path | None = None
        self._component_paths: tuple[Path, Path] | None = None

    def __enter__(self) -> tuple[Path, Path]:
        if self._temporary_dir is not None:
            raise RuntimeError("model bundle is already open")

        key = self._key_path.read_bytes()
        if len(key) != 32:
            raise ValueError("model bundle key must contain exactly 32 bytes")

        payload = memoryview(self._bundle_path.read_bytes())
        header_size = _HEADER.size + (_ENTRY.size * _COMPONENT_COUNT)
        if len(payload) < header_size:
            raise ValueError("model bundle is truncated")

        magic, version, component_count, reserved = _HEADER.unpack_from(payload, 0)
        if (
            magic != _MAGIC
            or version != _VERSION
            or component_count != _COMPONENT_COUNT
            or reserved != 0
        ):
            raise ValueError("unsupported model bundle format")

        entries: list[tuple[bytes, int, bytes]] = []
        cursor = _HEADER.size
        for _ in range(_COMPONENT_COUNT):
            nonce, encrypted_size, digest = _ENTRY.unpack_from(payload, cursor)
            entries.append((nonce, encrypted_size, digest))
            cursor += _ENTRY.size

        encrypted_total = sum(encrypted_size for _, encrypted_size, _ in entries)
        if cursor + encrypted_total != len(payload):
            raise ValueError("model bundle size does not match its header")

        temporary_root = _select_temporary_root()
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=".ms-model-", dir=str(temporary_root))
        )
        temporary_dir.chmod(0o700)
        self._temporary_dir = temporary_dir

        component_paths: list[Path] = []
        decryptor = AESGCM(key)
        try:
            for index, (nonce, encrypted_size, digest) in enumerate(entries):
                encrypted = bytes(payload[cursor : cursor + encrypted_size])
                cursor += encrypted_size
                associated_data = _MAGIC + bytes((_VERSION, index))
                plain = decryptor.decrypt(nonce, encrypted, associated_data)
                if not hmac.compare_digest(hashlib.sha256(plain).digest(), digest):
                    raise ValueError(f"model component {index} failed integrity validation")

                component_path = temporary_dir / f"component-{index}.engine"
                file_descriptor = os.open(
                    component_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                try:
                    with os.fdopen(file_descriptor, "wb", closefd=True) as stream:
                        metadata = json.dumps(
                            _ENGINE_METADATA[index],
                            separators=(",", ":"),
                        ).encode("utf-8")
                        stream.write(len(metadata).to_bytes(4, byteorder="little"))
                        stream.write(metadata)
                        stream.write(plain)
                        stream.flush()
                        os.fsync(stream.fileno())
                except Exception:
                    try:
                        os.close(file_descriptor)
                    except OSError:
                        pass
                    raise
                component_paths.append(component_path)
        except Exception:
            self.close()
            raise

        self._component_paths = (component_paths[0], component_paths[1])
        return self._component_paths

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._component_paths = None
        if self._temporary_dir is not None:
            shutil.rmtree(self._temporary_dir, ignore_errors=True)
            self._temporary_dir = None


def resolve_model_bundle_key_path(value: str | Path | None) -> Path:
    if value:
        return Path(value)
    environment_value = os.environ.get("MAIXSENSE_MODEL_KEY_PATH")
    if environment_value:
        return Path(environment_value)
    raise ValueError(
        "model bundle key path is required through --model-bundle-key-path "
        "or MAIXSENSE_MODEL_KEY_PATH"
    )


def _select_temporary_root() -> Path:
    configured = os.environ.get("MAIXSENSE_MODEL_TMPDIR")
    if configured:
        root = Path(configured)
    else:
        shared_memory = Path("/dev/shm")
        root = shared_memory if shared_memory.is_dir() else Path(tempfile.gettempdir())
    if not root.is_dir():
        raise ValueError(f"model temporary directory does not exist: {root}")
    return root
