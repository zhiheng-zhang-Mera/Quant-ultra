"""Shared enums, canonical serialization and validation primitives."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Mapping


class LifecycleStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASS = "PASS"
    HOLD = "HOLD"
    REJECT = "REJECT"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class AdmissionAction(str, Enum):
    PROCEED = "PROCEED"
    REFORMULATE = "REFORMULATE"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    OBSERVATION_ONLY = "OBSERVATION_ONLY"
    PRODUCTION_CANDIDATE = "PRODUCTION_CANDIDATE"
    ACCEPT_PRODUCTION = "ACCEPT_PRODUCTION"
    REJECTED = "REJECTED"
    HOLD_FOR_REVIEW = "HOLD_FOR_REVIEW"


class LockState(str, Enum):
    DRAFT = "DRAFT"
    PREREGISTERED = "PREREGISTERED"
    LOCKED = "LOCKED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    INVALIDATED = "INVALIDATED"


class Permission(str, Enum):
    RESEARCH = "RESEARCH"
    IMPLEMENTATION = "IMPLEMENTATION"
    VALIDATION = "VALIDATION"
    GOVERNANCE = "GOVERNANCE"
    PRODUCTION = "PRODUCTION"


class ValidationError(ValueError):
    """Raised when untrusted input violates a Research OS contract."""


ID_PREFIXES = frozenset({"REQ", "HYP", "EXP", "SRC", "EVD", "RUN", "MEM", "AGT"})
_ID_RE = re.compile(r"^(REQ|HYP|EXP|SRC|EVD|RUN|MEM|AGT)-[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_stable_id(value: str, expected_prefix: str | None = None) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ValidationError(f"invalid stable id: {value!r}")
    if expected_prefix and not value.startswith(expected_prefix + "-"):
        raise ValidationError(f"id must use {expected_prefix}- prefix")
    return value


def stable_id(prefix: str, *parts: Any) -> str:
    if prefix not in ID_PREFIXES:
        raise ValidationError(f"unknown id prefix: {prefix}")
    digest = hashlib.sha256(canonical_bytes(list(parts))).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValidationError("naive datetimes are not canonical")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if is_dataclass(value):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=lambda item: canonical_json(item))
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ValidationError("non-finite floats are not canonical")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValidationError(f"unsupported canonical type: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class Contract:
    """Immutable base for persistent objects with deterministic serialization."""

    schema_version: str
    SCHEMA: ClassVar[str] = "research-os/contract/v1"

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA:
            raise ValidationError(f"expected schema_version={self.SCHEMA!r}")

    def to_dict(self) -> dict[str, Any]:
        return _canonical(asdict(self))

    def to_json(self) -> str:
        return canonical_json(self)

    @property
    def content_sha256(self) -> str:
        return sha256(self)

    @property
    def record_sha256(self) -> str:
        """Hash the complete contract even when a subtype has a source-content field."""
        return sha256(self)
