"""Content-addressed dataset lineage from source through experiment input."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class DatasetLineage:
    dataset_id: str
    source_name: str
    source_uri: str
    license: str
    downloaded_at: str
    coverage_start: str
    coverage_end: str
    raw_path: str
    cleaned_path: str
    transformations: tuple[str, ...]
    anomalies: tuple[str, ...] = ()
    predecessor_version: str | None = None
    source_replacement_reason: str | None = None

    def validate(self) -> None:
        if not all((self.dataset_id, self.source_name, self.source_uri, self.license, self.downloaded_at,
                    self.coverage_start, self.coverage_end, self.raw_path, self.cleaned_path, self.transformations)):
            raise ValueError("dataset lineage requires source, coverage, files and transformations")
        if not pd_timestamp(self.coverage_start) <= pd_timestamp(self.coverage_end):
            raise ValueError("dataset coverage is reversed")
        if self.source_replacement_reason and not self.predecessor_version:
            raise ValueError("source replacement requires a predecessor version")


def pd_timestamp(value: str):
    from pandas import Timestamp
    return Timestamp(value)


def build_dataset_manifest(lineages: Iterable[DatasetLineage]) -> dict:
    rows = []
    for lineage in lineages:
        lineage.validate()
        raw, cleaned = Path(lineage.raw_path), Path(lineage.cleaned_path)
        if not raw.is_file() or not cleaned.is_file():
            raise FileNotFoundError(f"lineage files missing for {lineage.dataset_id}")
        row = {**asdict(lineage), "raw_sha256": sha256_file(raw), "cleaned_sha256": sha256_file(cleaned),
               "raw_bytes": raw.stat().st_size, "cleaned_bytes": cleaned.stat().st_size}
        row["dataset_version"] = hashlib.sha256(_canonical(row).encode("utf-8")).hexdigest()
        rows.append(row)
    if not rows:
        raise ValueError("at least one dataset lineage is required")
    payload = {"schema_version": "data-provenance/v1", "datasets": sorted(rows, key=lambda row: row["dataset_id"])}
    payload["manifest_sha256"] = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return payload


def verify_dataset_manifest(manifest: dict) -> dict:
    failures = []
    for row in manifest.get("datasets", []):
        for field, hash_field in (("raw_path", "raw_sha256"), ("cleaned_path", "cleaned_sha256")):
            path = Path(row.get(field, ""))
            if not path.is_file():
                failures.append(f"missing:{row.get('dataset_id')}:{field}")
            elif sha256_file(path) != row.get(hash_field):
                failures.append(f"hash_mismatch:{row.get('dataset_id')}:{field}")
    material = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if hashlib.sha256(_canonical(material).encode("utf-8")).hexdigest() != manifest.get("manifest_sha256"):
        failures.append("manifest_hash_mismatch")
    return {"valid": not failures, "failures": failures, "datasets": len(manifest.get("datasets", []))}


def append_provenance_event(path: Path, manifest: dict, event_type: str = "CREATED") -> dict:
    path = Path(path)
    prior = verify_provenance_history(path) if path.exists() else {"valid": True, "events": [], "head": "GENESIS"}
    if not prior["valid"]:
        raise ValueError("existing provenance history is invalid")
    event = {"schema_version": "data-provenance-event/v1", "event_type": event_type,
             "recorded_at": datetime.now(timezone.utc).isoformat(), "previous_hash": prior["head"],
             "manifest_sha256": manifest.get("manifest_sha256"),
             "dataset_versions": {row["dataset_id"]: row["dataset_version"] for row in manifest.get("datasets", [])}}
    event["event_hash"] = hashlib.sha256(_canonical(event).encode("utf-8")).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_canonical(event) + "\n")
    return event


def verify_provenance_history(path: Path) -> dict:
    events = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    previous, failures = "GENESIS", []
    for index, event in enumerate(events):
        claimed = event.get("event_hash")
        material = {key: value for key, value in event.items() if key != "event_hash"}
        actual = hashlib.sha256(_canonical(material).encode("utf-8")).hexdigest()
        if event.get("previous_hash") != previous or claimed != actual:
            failures.append(index)
        previous = claimed
    return {"valid": not failures, "failures": failures, "events": events, "head": previous}


def build_result_lineage(data_manifest: dict, *, feature_version: str, experiment_id: str,
                         code_version: str, parameter_version: str, result_sha256: str) -> dict:
    if not all((feature_version, experiment_id, code_version, parameter_version, result_sha256)):
        raise ValueError("result lineage requires data, features, experiment, code, parameters and result hash")
    return {"schema_version": "result-lineage/v1", "source_to_result": {
        "data_manifest": data_manifest["manifest_sha256"], "feature_version": feature_version,
        "experiment_id": experiment_id, "code_version": code_version,
        "parameter_version": parameter_version, "result_sha256": result_sha256,
    }}
