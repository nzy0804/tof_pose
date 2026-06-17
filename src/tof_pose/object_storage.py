from __future__ import annotations

import os
import posixpath
import re
from dataclasses import dataclass
from typing import Protocol


_SAFE_PART_RE = re.compile(r"[^A-Za-z0-9._=-]+")


def _first_value(*values: str | None) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


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
        )


class AliyunOSSClient:
    def __init__(self, config: ObjectStorageConfig) -> None:
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
        if session is not None:
            self._bucket = oss2.Bucket(auth, config.endpoint, config.bucket, session=session)
        else:
            self._bucket = oss2.Bucket(auth, config.endpoint, config.bucket)

    def get_bytes(self, object_key: str) -> bytes:
        return self._bucket.get_object(object_key).read()

    def put_bytes(self, object_key: str, data: bytes, content_type: str | None = None) -> str:
        headers = {}
        if content_type:
            headers["Content-Type"] = content_type
        self._bucket.put_object(object_key, data, headers=headers or None)
        return object_key


class S3ObjectStorageClient:
    def __init__(self, config: ObjectStorageConfig) -> None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError("S3-compatible object storage support requires the boto3 package") from exc

        self._bucket = config.bucket
        kwargs = {
            "service_name": "s3",
            "endpoint_url": config.endpoint,
            "aws_access_key_id": config.access_key_id,
            "aws_secret_access_key": config.access_key_secret,
            "config": Config(max_pool_connections=config.max_pool_connections),
        }
        if config.region:
            kwargs["region_name"] = config.region
        if config.security_token:
            kwargs["aws_session_token"] = config.security_token
        self._client = boto3.client(**kwargs)

    def get_bytes(self, object_key: str) -> bytes:
        obj = self._client.get_object(Bucket=self._bucket, Key=object_key)
        return obj["Body"].read()

    def put_bytes(self, object_key: str, data: bytes, content_type: str | None = None) -> str:
        kwargs = {
            "Bucket": self._bucket,
            "Key": object_key,
            "Body": data,
        }
        if content_type:
            kwargs["ContentType"] = content_type
        self._client.put_object(**kwargs)
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


def build_result_object_key(
    *,
    output_prefix: str,
    device_id: str,
    batch_id: str,
    frame_id: str,
    output_index: int,
    result_kind: str,
    image_name: str,
    image_format: str,
) -> str:
    prefix = str(output_prefix or "ai-results").strip("/")
    parts = [
        prefix,
        sanitize_object_key_part(device_id, "device"),
        sanitize_object_key_part(batch_id, "batch"),
        (
            f"{max(0, int(output_index)):04d}_"
            f"{sanitize_object_key_part(result_kind, 'result')}_"
            f"{sanitize_object_key_part(frame_id, 'frame')}_"
            f"{sanitize_object_key_part(image_name, 'image')}."
            f"{image_extension(image_format)}"
        ),
    ]
    return posixpath.join(*[part for part in parts if part])
