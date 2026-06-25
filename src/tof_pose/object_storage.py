from __future__ import annotations

import os
import posixpath
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from typing import Protocol


LOGGER = logging.getLogger(__name__)
_SAFE_PART_RE = re.compile(r"[^A-Za-z0-9._=-]+")
_OBJECT_KEY_TIMEZONE = timezone(timedelta(hours=8))


def _first_value(*values: str | None) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _optional_float(value: str | float | int | None, default: float | None) -> float | None:
    text = str(value if value is not None else "").strip()
    if not text:
        return default
    try:
        parsed = float(text)
    except ValueError:
        return default
    return parsed if parsed > 0 else None


def _optional_int(value: str | int | None, default: int) -> int:
    text = str(value if value is not None else "").strip()
    if not text:
        return default
    try:
        return max(0, int(text))
    except ValueError:
        return default


class ObjectStorageOperationError(RuntimeError):
    pass


class ObjectStorageClient(Protocol):
    def get_bytes(self, object_key: str) -> bytes:
        ...

    def put_bytes(self, object_key: str, data: bytes, content_type: str | None = None) -> str:
        ...


@dataclass(frozen=True)
class ObjectStorageConfig:
    provider: str
    endpoint: str
    bucket: str
    access_key_id: str
    access_key_secret: str
    region: str | None = None
    security_token: str | None = None
    output_prefix: str = "ai-results"
    max_pool_connections: int = 128
    connect_timeout_sec: float | None = 1.0
    read_timeout_sec: float | None = 2.0
    request_retries: int = 2
    request_deadline_sec: float | None = 3.0
    retry_backoff_ms: int = 100

    @classmethod
    def from_values(
        cls,
        *,
        provider: str | None = None,
        endpoint: str | None = None,
        bucket: str | None = None,
        access_key_id: str | None = None,
        access_key_secret: str | None = None,
        region: str | None = None,
        security_token: str | None = None,
        output_prefix: str | None = None,
        max_pool_connections: int | None = None,
        connect_timeout_sec: float | None = None,
        read_timeout_sec: float | None = None,
        request_retries: int | None = None,
        request_deadline_sec: float | None = None,
        retry_backoff_ms: int | None = None,
    ) -> "ObjectStorageConfig | None":
        provider_value = _first_value(provider, os.environ.get("OSS_PROVIDER")).lower()
        endpoint_value = _first_value(endpoint, os.environ.get("OSS_ENDPOINT"), os.environ.get("S3_ENDPOINT_URL"))
        bucket_value = _first_value(bucket, os.environ.get("OSS_BUCKET"), os.environ.get("S3_BUCKET"), os.environ.get("MINIO_BUCKET"))
        access_key_id_value = _first_value(
            access_key_id,
            os.environ.get("OSS_ACCESS_KEY_ID"),
            os.environ.get("AWS_ACCESS_KEY_ID"),
            os.environ.get("MINIO_ACCESS_KEY"),
        )
        access_key_secret_value = _first_value(
            access_key_secret,
            os.environ.get("OSS_ACCESS_KEY_SECRET"),
            os.environ.get("AWS_SECRET_ACCESS_KEY"),
            os.environ.get("MINIO_SECRET_KEY"),
        )
        region_value = _first_value(region, os.environ.get("OSS_REGION"), os.environ.get("AWS_REGION")) or None
        security_token_value = _first_value(security_token, os.environ.get("OSS_SECURITY_TOKEN"), os.environ.get("AWS_SESSION_TOKEN")) or None
        output_prefix_value = _first_value(output_prefix, os.environ.get("OSS_OUTPUT_PREFIX"), "ai-results")
        max_pool_connections_value = max_pool_connections
        if max_pool_connections_value is None:
            try:
                max_pool_connections_value = int(os.environ.get("OSS_MAX_POOL_CONNECTIONS") or "128")
            except ValueError:
                max_pool_connections_value = 128
        connect_timeout_value = _optional_float(
            connect_timeout_sec,
            _optional_float(os.environ.get("OSS_CONNECT_TIMEOUT_SEC"), 1.0),
        )
        read_timeout_value = _optional_float(
            read_timeout_sec,
            _optional_float(os.environ.get("OSS_READ_TIMEOUT_SEC"), 2.0),
        )
        request_deadline_value = _optional_float(
            request_deadline_sec,
            _optional_float(os.environ.get("OSS_REQUEST_DEADLINE_SEC"), 3.0),
        )
        request_retries_value = _optional_int(
            request_retries,
            _optional_int(os.environ.get("OSS_REQUEST_RETRIES"), 2),
        )
        retry_backoff_value = _optional_int(
            retry_backoff_ms,
            _optional_int(os.environ.get("OSS_RETRY_BACKOFF_MS"), 100),
        )

        if not any((provider_value, endpoint_value, bucket_value, access_key_id_value, access_key_secret_value)):
            return None
        if not provider_value:
            provider_value = "aliyun"
        missing = [
            name
            for name, value in (
                ("oss_endpoint", endpoint_value),
                ("oss_bucket", bucket_value),
                ("oss_access_key_id", access_key_id_value),
                ("oss_access_key_secret", access_key_secret_value),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"object storage config missing: {', '.join(missing)}")
        if provider_value not in {"aliyun", "s3"}:
            raise ValueError("oss_provider must be aliyun or s3")

        return cls(
            provider=provider_value,
            endpoint=endpoint_value,
            bucket=bucket_value,
            access_key_id=access_key_id_value,
            access_key_secret=access_key_secret_value,
            region=region_value,
            security_token=security_token_value,
            output_prefix=output_prefix_value.strip("/"),
            max_pool_connections=max(1, int(max_pool_connections_value)),
            connect_timeout_sec=connect_timeout_value,
            read_timeout_sec=read_timeout_value,
            request_retries=request_retries_value,
            request_deadline_sec=request_deadline_value,
            retry_backoff_ms=retry_backoff_value,
        )


def _timeout_value(config: ObjectStorageConfig):
    if config.connect_timeout_sec is not None and config.read_timeout_sec is not None:
        return (float(config.connect_timeout_sec), float(config.read_timeout_sec))
    if config.connect_timeout_sec is not None:
        return float(config.connect_timeout_sec)
    if config.read_timeout_sec is not None:
        return float(config.read_timeout_sec)
    return None


class _RetryingObjectStorageClient:
    def __init__(self, config: ObjectStorageConfig) -> None:
        self._request_retries = max(0, int(config.request_retries))
        self._request_deadline_sec = (
            float(config.request_deadline_sec)
            if config.request_deadline_sec is not None and float(config.request_deadline_sec) > 0
            else None
        )
        self._retry_backoff_sec = max(0.0, float(config.retry_backoff_ms) / 1000.0)

    def _run_with_retries(self, operation: str, object_key: str, call):
        attempts = max(1, self._request_retries + 1)
        started = time.monotonic()
        deadline_at = started + self._request_deadline_sec if self._request_deadline_sec is not None else None
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            if deadline_at is not None and time.monotonic() >= deadline_at:
                break
            try:
                return call()
            except Exception as exc:
                last_exc = exc
                if attempt >= attempts:
                    break
                if deadline_at is not None and time.monotonic() >= deadline_at:
                    break
                sleep_sec = self._retry_backoff_sec * (2 ** (attempt - 1))
                if deadline_at is not None:
                    remaining = max(0.0, deadline_at - time.monotonic())
                    sleep_sec = min(sleep_sec, remaining)
                LOGGER.warning(
                    "Object storage %s retry: attempt=%d/%d key=%s error=%s",
                    operation,
                    attempt,
                    attempts,
                    object_key,
                    type(exc).__name__,
                )
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        raise ObjectStorageOperationError(
            f"object storage {operation} failed after {attempts} attempts in {elapsed_ms}ms for key={object_key}"
        ) from last_exc


class AliyunOSSClient(_RetryingObjectStorageClient):
    def __init__(self, config: ObjectStorageConfig) -> None:
        super().__init__(config)
        try:
            import oss2
        except ImportError as exc:
            raise RuntimeError("aliyun OSS support requires the oss2 package") from exc

        if config.security_token:
            auth = oss2.StsAuth(config.access_key_id, config.access_key_secret, config.security_token)
        else:
            auth = oss2.Auth(config.access_key_id, config.access_key_secret)
        session = None
        if hasattr(oss2, "Session"):
            try:
                session = oss2.Session(pool_size=config.max_pool_connections)
            except TypeError:
                session = oss2.Session()
        timeout = _timeout_value(config)
        kwargs = {}
        if timeout is not None:
            kwargs["connect_timeout"] = timeout
        if session is not None:
            self._bucket = oss2.Bucket(auth, config.endpoint, config.bucket, session=session, **kwargs)
        else:
            self._bucket = oss2.Bucket(auth, config.endpoint, config.bucket, **kwargs)

    def get_bytes(self, object_key: str) -> bytes:
        return self._run_with_retries("download", object_key, lambda: self._bucket.get_object(object_key).read())

    def put_bytes(self, object_key: str, data: bytes, content_type: str | None = None) -> str:
        headers = {}
        if content_type:
            headers["Content-Type"] = content_type
        self._run_with_retries("upload", object_key, lambda: self._bucket.put_object(object_key, data, headers=headers or None))
        return object_key


class S3ObjectStorageClient(_RetryingObjectStorageClient):
    def __init__(self, config: ObjectStorageConfig) -> None:
        super().__init__(config)
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError("S3-compatible object storage support requires the boto3 package") from exc

        self._bucket = config.bucket
        client_config_kwargs = {
            "max_pool_connections": config.max_pool_connections,
        }
        if config.connect_timeout_sec is not None:
            client_config_kwargs["connect_timeout"] = float(config.connect_timeout_sec)
        if config.read_timeout_sec is not None:
            client_config_kwargs["read_timeout"] = float(config.read_timeout_sec)
        kwargs = {
            "service_name": "s3",
            "endpoint_url": config.endpoint,
            "aws_access_key_id": config.access_key_id,
            "aws_secret_access_key": config.access_key_secret,
            "config": Config(**client_config_kwargs),
        }
        if config.region:
            kwargs["region_name"] = config.region
        if config.security_token:
            kwargs["aws_session_token"] = config.security_token
        self._client = boto3.client(**kwargs)

    def get_bytes(self, object_key: str) -> bytes:
        def call():
            obj = self._client.get_object(Bucket=self._bucket, Key=object_key)
            return obj["Body"].read()

        return self._run_with_retries("download", object_key, call)

    def put_bytes(self, object_key: str, data: bytes, content_type: str | None = None) -> str:
        kwargs = {
            "Bucket": self._bucket,
            "Key": object_key,
            "Body": data,
        }
        if content_type:
            kwargs["ContentType"] = content_type
        self._run_with_retries("upload", object_key, lambda: self._client.put_object(**kwargs))
        return object_key


def create_object_storage_client(config: ObjectStorageConfig | None) -> ObjectStorageClient | None:
    if config is None:
        return None
    if config.provider == "aliyun":
        return AliyunOSSClient(config)
    if config.provider == "s3":
        return S3ObjectStorageClient(config)
    raise ValueError("unsupported object storage provider")


def sanitize_object_key_part(value: str | int | None, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = _SAFE_PART_RE.sub("_", text)
    return text.strip("._-/") or fallback


def image_content_type(image_format: str | None) -> str:
    normalized = str(image_format or "").strip().lower()
    if normalized in {"jpg", "jpeg"}:
        return "image/jpeg"
    if normalized == "png":
        return "image/png"
    return "application/octet-stream"


def image_extension(image_format: str | None) -> str:
    normalized = str(image_format or "").strip().lower()
    if normalized in {"jpg", "jpeg"}:
        return "jpg"
    if normalized == "png":
        return "png"
    return "bin"


def object_key_date(timestamp_ms: int | None) -> str:
    try:
        timestamp = max(0, int(timestamp_ms or 0))
    except (TypeError, ValueError):
        timestamp = 0
    return datetime.fromtimestamp(timestamp / 1000.0, _OBJECT_KEY_TIMEZONE).strftime("%Y%m%d")


def build_result_object_key(
    *,
    output_prefix: str,
    device_id: str,
    timestamp_ms: int,
    image_name: str,
    image_format: str,
) -> str:
    _ = output_prefix
    try:
        timestamp = max(0, int(timestamp_ms or 0))
    except (TypeError, ValueError):
        timestamp = 0
    parts = [
        sanitize_object_key_part(device_id, "device"),
        object_key_date(timestamp),
        (
            f"{timestamp}_"
            f"{sanitize_object_key_part(image_name, 'image')}."
            f"{image_extension(image_format)}"
        ),
    ]
    return posixpath.join(*[part for part in parts if part])
