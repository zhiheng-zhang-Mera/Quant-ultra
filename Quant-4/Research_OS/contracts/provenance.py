"""Source snapshot and Git-less materialized workspace provenance."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar

from .common import Contract, utc_now


@dataclass(frozen=True)
class SourceSnapshotManifest(Contract):
    SCHEMA: ClassVar[str] = "source-snapshot-manifest/v1"
    source_git_sha: str = ""
    source_git_state: str = "UNKNOWN"
    source_tree_hash: str = ""
    dependency_lock_hashes: dict[str, str] = field(default_factory=dict)
    spec_hash: str = ""
    data_manifest_hash: str = ""
    captured_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class MaterializedWorkspaceManifest(Contract):
    SCHEMA: ClassVar[str] = "materialized-workspace-manifest/v1"
    workspace_id: str = ""
    workspace_content_tree_hash: str = ""
    source_snapshot_hash: str = ""
    shared_writable_cache: bool = False
    excluded_paths: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=utc_now)
