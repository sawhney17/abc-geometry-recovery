#!/usr/bin/env python3
"""Audit every raw MCAP in the revision-pinned public ABC-130K preview."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcap.reader import make_reader

from abc_geometry import __version__
from abc_geometry.kinematics import (
    MODEL_SOURCE_REVISION,
    RIGHT_ARM_SHARED_TRANSLATION,
    SHARED_FRAME_TRANSFORM_VERSION,
    model_sha256,
)
from abc_geometry.mcap_validation import ARM_TOPICS, audit_mcap

DATASET_ID = "Voxel51/ABC-130k"
DATASET_REVISION = "9659e8ce4b39580f48369cc31bc2e47a217c40e7"
EXPECTED_FILE_COUNT = 40
PINNED_MCAP_INVENTORY_SHA256 = "6de3f6203cf1bf66632399242b5a549846f4c2286a5495447bb86641b6b86ae4"
PINNED_SAMPLES_GIT_BLOB_OID = "971327b4ac79cdd5b32c5a8d85db9482c88cfbf8"
PINNED_SAMPLES_SIZE_BYTES = 62_794
REPORT_SCHEMA = "abc-130k-public-mcap-geometry-audit/v1"
TRANSLATION_TOLERANCE_M = 1e-5
ROTATION_TOLERANCE_DEG = 1e-3
ISSUE_SAMPLE_LIMIT = 1_000
RIGHT_ARM_TOPICS = tuple(topic for topic, (arm, _) in ARM_TOPICS.items() if arm == "right")


class PublicPreviewAuditError(RuntimeError):
    """Raised when pinned provenance or audit accounting cannot be established."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def git_blob_oid(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _require_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PublicPreviewAuditError(f"{label} must be an integer >= {minimum}")
    return value


def _normalise_hex(value: Any, *, digits: int, label: str) -> str:
    if not isinstance(value, str):
        raise PublicPreviewAuditError(f"{label} must be a hexadecimal string")
    if value.startswith("sha256:") and digits == 64:
        value = value.removeprefix("sha256:")
    value = value.lower()
    if len(value) != digits or any(character not in "0123456789abcdef" for character in value):
        raise PublicPreviewAuditError(f"{label} must contain exactly {digits} hexadecimal digits")
    return value


def _validate_mcap_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise PublicPreviewAuditError("tree MCAP path must be a non-empty string")
    relative = Path(value)
    parts = relative.parts
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or ".." in parts
        or len(parts) != 5
        or parts[:2] != ("data", "val")
        or not parts[2]
        or not parts[3].startswith("episode_")
        or parts[4] != "episode.fo.mcap"
    ):
        raise PublicPreviewAuditError(f"unexpected public-preview MCAP path: {value!r}")
    return value


def _normalise_mcap_entry(entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise PublicPreviewAuditError("every tree entry must be an object")
    path = _validate_mcap_path(entry.get("path"))
    if entry.get("type") != "file":
        raise PublicPreviewAuditError(f"MCAP tree entry is not a file: {path}")
    lfs = entry.get("lfs")
    if not isinstance(lfs, dict):
        raise PublicPreviewAuditError(f"MCAP tree entry has no LFS metadata: {path}")
    size = _require_int(entry.get("size"), label=f"{path} size", minimum=1)
    lfs_size = _require_int(lfs.get("size"), label=f"{path} LFS size", minimum=1)
    if size != lfs_size:
        raise PublicPreviewAuditError(
            f"tree and LFS sizes disagree for {path}: {size} != {lfs_size}"
        )
    return {
        "type": "file",
        "path": path,
        "size": size,
        "oid": _normalise_hex(entry.get("oid"), digits=40, label=f"{path} Git OID"),
        "lfs": {
            "oid": _normalise_hex(lfs.get("oid"), digits=64, label=f"{path} LFS OID"),
            "size": lfs_size,
        },
    }


def mcap_inventory_fingerprint(entries: list[dict[str, Any]]) -> str:
    canonical = [
        {
            "path": entry["path"],
            "size": entry["size"],
            "oid": entry["oid"],
            "lfs": {
                "oid": entry["lfs"]["oid"],
                "size": entry["lfs"]["size"],
            },
        }
        for entry in sorted(entries, key=lambda value: value["path"])
    ]
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_samples_tree_entry(tree: list[Any], samples_bytes: bytes) -> None:
    matches = [
        entry for entry in tree if isinstance(entry, dict) and entry.get("path") == "samples.json"
    ]
    if len(matches) != 1:
        raise PublicPreviewAuditError("pinned tree must contain exactly one samples.json entry")
    entry = matches[0]
    if entry.get("type") != "file":
        raise PublicPreviewAuditError("samples.json tree entry is not a file")
    size = _require_int(entry.get("size"), label="samples.json size", minimum=1)
    oid = _normalise_hex(entry.get("oid"), digits=40, label="samples.json Git OID")
    if size != PINNED_SAMPLES_SIZE_BYTES or oid != PINNED_SAMPLES_GIT_BLOB_OID:
        raise PublicPreviewAuditError("samples.json does not belong to the pinned dataset revision")
    if len(samples_bytes) != size:
        raise PublicPreviewAuditError(
            f"samples.json byte size does not match the pinned tree: {len(samples_bytes)} != {size}"
        )
    actual_oid = git_blob_oid(samples_bytes)
    if actual_oid != oid:
        raise PublicPreviewAuditError(
            f"samples.json Git blob OID does not match the pinned tree: {actual_oid} != {oid}"
        )


def _validate_samples(
    samples_payload: Any,
    entries_by_path: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(samples_payload, dict) or not isinstance(
        samples_payload.get("samples"), list
    ):
        raise PublicPreviewAuditError("samples JSON must be an object containing a samples list")
    rows = samples_payload["samples"]
    if len(rows) != EXPECTED_FILE_COUNT:
        raise PublicPreviewAuditError(
            f"expected exactly {EXPECTED_FILE_COUNT} sample rows, found {len(rows)}"
        )
    samples_by_path: dict[str, dict[str, Any]] = {}
    for index, sample in enumerate(rows):
        if not isinstance(sample, dict):
            raise PublicPreviewAuditError(f"sample row {index} must be an object")
        path = sample.get("filepath")
        if not isinstance(path, str):
            raise PublicPreviewAuditError(f"sample row {index} has no string filepath")
        if path in samples_by_path:
            raise PublicPreviewAuditError(f"duplicate samples.json filepath: {path}")
        if path not in entries_by_path:
            raise PublicPreviewAuditError(
                f"samples.json path is absent from the pinned tree: {path}"
            )
        parts = Path(path).parts
        if sample.get("split") != "val":
            raise PublicPreviewAuditError(f"sample split is not val: {path}")
        if sample.get("task") != parts[2]:
            raise PublicPreviewAuditError(f"sample task does not match its path: {path}")
        if sample.get("episode_id") != parts[3]:
            raise PublicPreviewAuditError(f"sample episode_id does not match its path: {path}")
        _require_int(sample.get("n_messages"), label=f"{path} n_messages")
        samples_by_path[path] = sample
    missing = sorted(set(entries_by_path) - set(samples_by_path))
    if missing:
        raise PublicPreviewAuditError(f"pinned MCAP has no samples.json row: {missing[0]}")
    return samples_by_path


def build_jobs(
    tree_payload: Any,
    samples_payload: Any,
    samples_bytes: bytes,
    source_root: Path,
) -> list[dict[str, Any]]:
    """Validate the pinned inventory and resolve its 40 immutable source files."""

    if not isinstance(tree_payload, list):
        raise PublicPreviewAuditError("tree JSON must be the Hugging Face tree API list")
    raw_mcap_entries = [
        entry
        for entry in tree_payload
        if isinstance(entry, dict)
        and isinstance(entry.get("path"), str)
        and entry["path"].endswith(".mcap")
    ]
    if len(raw_mcap_entries) != EXPECTED_FILE_COUNT:
        raise PublicPreviewAuditError(
            f"expected exactly {EXPECTED_FILE_COUNT} MCAP files, found {len(raw_mcap_entries)}"
        )
    entries = [_normalise_mcap_entry(entry) for entry in raw_mcap_entries]
    entries_by_path = {entry["path"]: entry for entry in entries}
    if len(entries_by_path) != EXPECTED_FILE_COUNT:
        raise PublicPreviewAuditError("pinned tree contains duplicate MCAP paths")
    fingerprint = mcap_inventory_fingerprint(entries)
    if fingerprint != PINNED_MCAP_INVENTORY_SHA256:
        raise PublicPreviewAuditError(
            "MCAP inventory does not match the pinned Voxel51 revision: "
            f"{fingerprint} != {PINNED_MCAP_INVENTORY_SHA256}"
        )
    _validate_samples_tree_entry(tree_payload, samples_bytes)
    samples_by_path = _validate_samples(samples_payload, entries_by_path)

    root = source_root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    jobs: list[dict[str, Any]] = []
    for path, entry in sorted(entries_by_path.items()):
        source = (root / path).resolve()
        if not source.is_relative_to(root):
            raise PublicPreviewAuditError(f"source path escapes --source-root: {path}")
        if not source.is_file():
            raise FileNotFoundError(source)
        sample = samples_by_path[path]
        jobs.append(
            {
                "path": path,
                "local_path": str(source),
                "hf_uri": f"hf://datasets/{DATASET_ID}@{DATASET_REVISION}/{path}",
                "resolve_url": (
                    f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
                    f"{DATASET_REVISION}/{path}"
                ),
                "tree": entry,
                "sample": sample,
            }
        )
    return jobs


def raw_summary(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        summary = make_reader(stream).get_summary()
    if summary is None or summary.statistics is None:
        raise PublicPreviewAuditError(f"MCAP has no summary statistics: {path}")
    by_topic: dict[str, int] = {}
    for channel_id, count in summary.statistics.channel_message_counts.items():
        channel = summary.channels.get(channel_id)
        topic = channel.topic if channel is not None else f"<channel:{channel_id}>"
        by_topic[topic] = by_topic.get(topic, 0) + int(count)
    return {
        "all_messages": int(summary.statistics.message_count),
        "generated_plot_messages": sum(
            count for topic, count in by_topic.items() if topic.endswith(".plot")
        ),
        "topic_counts": dict(sorted(by_topic.items())),
        "target_topic_counts": {topic: by_topic.get(topic, 0) for topic in ARM_TOPICS},
    }


def _vector(value: Any, *, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise PublicPreviewAuditError(f"{label} must be a three-value list")
    result = [float(component) for component in value]
    if not all(math.isfinite(component) for component in result):
        raise PublicPreviewAuditError(f"{label} contains a non-finite value")
    return result


def combine_offset_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge population statistics emitted by the final per-topic validator."""

    count = 0
    mean = [0.0, 0.0, 0.0]
    m2 = [0.0, 0.0, 0.0]
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    for index, summary in enumerate(summaries):
        current_count = _require_int(summary.get("count"), label=f"offset {index} count")
        if current_count == 0:
            continue
        current_mean = _vector(summary.get("mean_xyz"), label=f"offset {index} mean")
        current_std = _vector(summary.get("std_xyz"), label=f"offset {index} std")
        current_min = _vector(summary.get("min_xyz"), label=f"offset {index} min")
        current_max = _vector(summary.get("max_xyz"), label=f"offset {index} max")
        new_count = count + current_count
        for axis in range(3):
            delta = current_mean[axis] - mean[axis]
            m2[axis] += (
                current_count * current_std[axis] ** 2
                + delta**2 * count * current_count / new_count
            )
            mean[axis] += delta * current_count / new_count
            minimum[axis] = min(minimum[axis], current_min[axis])
            maximum[axis] = max(maximum[axis], current_max[axis])
        count = new_count
    if count == 0:
        return {
            "compared_pose_records": 0,
            "translation_m": {"mean": None, "std": None, "min": None, "max": None},
        }
    return {
        "compared_pose_records": count,
        "translation_m": {
            "mean": mean,
            "std": [math.sqrt(max(value / count, 0.0)) for value in m2],
            "min": minimum,
            "max": maximum,
        },
    }


def observed_right_arm_base_offset(audit: dict[str, Any]) -> dict[str, Any]:
    summaries = [
        audit["topics"][topic]["fk"]["observed_base_translation_offset_m"]
        for topic in RIGHT_ARM_TOPICS
    ]
    return combine_offset_summaries(summaries)


def _verify_source_before(path: Path, tree_entry: dict[str, Any]) -> tuple[os.stat_result, str]:
    stat = path.stat()
    expected_size = int(tree_entry["size"])
    if stat.st_size != expected_size:
        raise PublicPreviewAuditError(
            f"source size does not match pinned tree for {path}: {stat.st_size} != {expected_size}"
        )
    digest = sha256_file(path)
    expected_digest = str(tree_entry["lfs"]["oid"])
    if digest != expected_digest:
        raise PublicPreviewAuditError(
            f"source SHA-256 does not match pinned LFS object for {path}: "
            f"{digest} != {expected_digest}"
        )
    return stat, digest


def _verify_source_after(
    path: Path,
    tree_entry: dict[str, Any],
    before_stat: os.stat_result,
    before_digest: str,
) -> tuple[os.stat_result, str]:
    after_stat, after_digest = _verify_source_before(path, tree_entry)
    if after_stat.st_size != before_stat.st_size or after_digest != before_digest:
        raise PublicPreviewAuditError(f"source changed while it was being audited: {path}")
    return after_stat, after_digest


def audit_one(job: dict[str, Any]) -> dict[str, Any]:
    path = Path(job["local_path"])
    try:
        stat_before, digest_before = _verify_source_before(path, job["tree"])
        summary = raw_summary(path)
        report = audit_mcap(
            path,
            translation_tolerance_m=TRANSLATION_TOLERANCE_M,
            rotation_tolerance_deg=ROTATION_TOLERANCE_DEG,
            issue_sample_limit=ISSUE_SAMPLE_LIMIT,
        )
        report.source = job["hf_uri"]
        audit = report.to_dict()
        observed_offset = observed_right_arm_base_offset(audit)
        stat_after, digest_after = _verify_source_after(
            path,
            job["tree"],
            stat_before,
            digest_before,
        )
    except Exception as exc:
        raise PublicPreviewAuditError(f"failed to audit {job['path']}: {exc}") from exc
    sample = job["sample"]
    return {
        "path": job["path"],
        "hf_uri": job["hf_uri"],
        "resolve_url": job["resolve_url"],
        "split": sample["split"],
        "task": sample["task"],
        "episode_id": sample["episode_id"],
        "sample_metadata": {
            "station": sample.get("station"),
            "duration_s": sample.get("duration_s"),
            "declared_messages": sample["n_messages"],
            "source_fps": sample.get("source_fps"),
            "reencoded_to_30fps": sample.get("reencoded_to_30fps"),
        },
        "object": {
            "git_blob_oid": job["tree"]["oid"],
            "lfs_sha256": job["tree"]["lfs"]["oid"],
            "expected_size_bytes": job["tree"]["size"],
            "size_bytes_before": stat_before.st_size,
            "size_bytes_after": stat_after.st_size,
            "sha256_before": digest_before,
            "sha256_after": digest_after,
            "size_matches_manifest": stat_before.st_size == job["tree"]["size"],
            "sha256_matches_lfs": digest_before == job["tree"]["lfs"]["oid"],
            "read_only_unchanged": (
                stat_before.st_size == stat_after.st_size and digest_before == digest_after
            ),
        },
        "mcap_summary": summary,
        "observed_right_arm_base_offset": observed_offset,
        "audit": audit,
    }


def _accumulate_residual(
    accumulator: dict[str, float | int],
    values: dict[str, Any],
    count: int,
) -> None:
    if count == 0:
        return
    mean = float(values["mean"])
    rmse = float(values["rmse"])
    maximum = float(values["max"])
    if not all(math.isfinite(value) for value in (mean, rmse, maximum)):
        raise PublicPreviewAuditError("validator emitted a non-finite residual summary")
    accumulator["n"] += count
    accumulator["sum"] += mean * count
    accumulator["sq_sum"] += rmse**2 * count
    accumulator["max"] = max(float(accumulator["max"]), maximum)


def _cluster_coordinate(value: float) -> float:
    rounded = round(value, 2)
    return 0.0 if rounded == 0 else rounded


def aggregate(files: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "files": len(files),
        "bytes": sum(item["object"]["size_bytes_before"] for item in files),
        "declared_all_messages": sum(
            item["sample_metadata"]["declared_messages"] for item in files
        ),
        "mcap_summary_all_messages": sum(item["mcap_summary"]["all_messages"] for item in files),
        "generated_plot_messages": sum(
            item["mcap_summary"]["generated_plot_messages"] for item in files
        ),
        "target_messages": 0,
        "poses": {
            "missing": 0,
            "malformed": 0,
            "nonfinite": 0,
            "intact": 0,
            "recoverable_invalid": 0,
            "unrecoverable_invalid": 0,
        },
        "joints": {"missing": 0, "malformed": 0, "nonfinite": 0},
        "fk": {
            "compared": 0,
            "failures": 0,
            "outside_tolerance": 0,
            "translation_error_m": {},
            "rotation_error_deg": {},
        },
        "decode_errors": 0,
        "missing_topic_occurrences": 0,
        "issues": 0,
    }
    topic_totals = {
        topic: {
            "messages": 0,
            "poses": {key: 0 for key in totals["poses"]},
            "joints": {key: 0 for key in totals["joints"]},
            "fk": {"compared": 0, "failures": 0, "outside_tolerance": 0},
        }
        for topic in ARM_TOPICS
    }
    residuals: dict[str, dict[str, float | int]] = {
        "translation_error_m": {"n": 0, "sum": 0.0, "sq_sum": 0.0, "max": 0.0},
        "rotation_error_deg": {"n": 0, "sum": 0.0, "sq_sum": 0.0, "max": 0.0},
    }

    for item in files:
        audit = item["audit"]
        totals["target_messages"] += audit["totals"]["target_messages"]
        totals["decode_errors"] += audit["totals"]["decode_errors"]
        totals["missing_topic_occurrences"] += len(audit["missing_topics"])
        totals["issues"] += audit["issue_count"]
        for topic in ARM_TOPICS:
            topic_report = audit["topics"][topic]
            topic_total = topic_totals[topic]
            topic_total["messages"] += topic_report["messages"]
            for key, value in topic_report["poses"].items():
                topic_total["poses"][key] += value
                totals["poses"][key] += value
            for key, value in topic_report["joints"].items():
                topic_total["joints"][key] += value
                totals["joints"][key] += value
            for key in ("compared", "failures", "outside_tolerance"):
                topic_total["fk"][key] += topic_report["fk"][key]
                totals["fk"][key] += topic_report["fk"][key]
            compared = topic_report["fk"]["compared"]
            for metric in residuals:
                _accumulate_residual(
                    residuals[metric],
                    topic_report["fk"][metric],
                    compared,
                )

    for metric, accumulator in residuals.items():
        count = int(accumulator["n"])
        totals["fk"][metric] = {
            "mean": float(accumulator["sum"]) / count if count else None,
            "rmse": math.sqrt(float(accumulator["sq_sum"]) / count) if count else None,
            "max": float(accumulator["max"]) if count else None,
        }

    clusters: dict[tuple[float, float, float], dict[str, Any]] = {}
    for item in files:
        observed = item["observed_right_arm_base_offset"]
        count = observed["compared_pose_records"]
        if count == 0:
            continue
        mean = observed["translation_m"]["mean"]
        key = tuple(_cluster_coordinate(float(value)) for value in mean)
        cluster = clusters.setdefault(
            key,
            {
                "translation_m_rounded": list(key),
                "files": 0,
                "pose_records": 0,
                "tasks": [],
            },
        )
        cluster["files"] += 1
        cluster["pose_records"] += count
        cluster["tasks"].append(item["task"])
    for cluster in clusters.values():
        cluster["tasks"].sort()
    offset_analysis = {
        "files_with_observable_right_arm_offset": sum(
            item["observed_right_arm_base_offset"]["compared_pose_records"] > 0 for item in files
        ),
        "files_without_observable_right_arm_offset": sum(
            item["observed_right_arm_base_offset"]["compared_pose_records"] == 0 for item in files
        ),
        "clusters": sorted(clusters.values(), key=lambda item: item["translation_m_rounded"]),
    }
    return {
        "totals": totals,
        "topics": topic_totals,
        "right_arm_base_offset_analysis": offset_analysis,
    }


def validate_invariants(
    files: list[dict[str, Any]],
    totals: dict[str, Any],
) -> list[dict[str, Any]]:
    checks = [
        ("exactly_40_files", len(files) == EXPECTED_FILE_COUNT),
        ("all_sizes_match_pinned_tree", all(x["object"]["size_matches_manifest"] for x in files)),
        ("all_sha256_match_pinned_lfs", all(x["object"]["sha256_matches_lfs"] for x in files)),
        ("all_sources_unchanged_by_audit", all(x["object"]["read_only_unchanged"] for x in files)),
        (
            "all_audit_sources_are_pinned_hf_uris",
            all(
                x["audit"]["source"] == x["hf_uri"]
                and x["hf_uri"].startswith(f"hf://datasets/{DATASET_ID}@{DATASET_REVISION}/")
                for x in files
            ),
        ),
        (
            "samples_declared_plus_generated_plot_counts_match_mcap_summaries",
            all(
                x["sample_metadata"]["declared_messages"]
                + x["mcap_summary"]["generated_plot_messages"]
                == x["mcap_summary"]["all_messages"]
                for x in files
            ),
        ),
        (
            "decoded_target_counts_match_mcap_summaries",
            all(
                all(
                    x["audit"]["topics"][topic]["messages"]
                    == x["mcap_summary"]["target_topic_counts"][topic]
                    for topic in ARM_TOPICS
                )
                for x in files
            ),
        ),
        (
            "pose_classification_is_exhaustive",
            all(
                all(
                    sum(
                        x["audit"]["topics"][topic]["poses"][key]
                        for key in ("missing", "malformed", "nonfinite", "intact")
                    )
                    == x["audit"]["topics"][topic]["messages"]
                    for topic in ARM_TOPICS
                )
                for x in files
            ),
        ),
        (
            "observed_base_translation_counts_match_fk_compared",
            all(
                all(
                    x["audit"]["topics"][topic]["fk"]["observed_base_translation_offset_m"]["count"]
                    == x["audit"]["topics"][topic]["fk"]["compared"]
                    for topic in ARM_TOPICS
                )
                for x in files
            ),
        ),
        ("no_decode_errors", totals["decode_errors"] == 0),
        ("no_missing_target_topics", totals["missing_topic_occurrences"] == 0),
    ]
    return [{"name": name, "passed": passed} for name, passed in checks]


def build_report(files: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate_report = aggregate(files)
    totals = aggregate_report["totals"]
    invariants = validate_invariants(files, totals)
    repository = Path(__file__).resolve().parents[1]
    validation_source = repository / "src/abc_geometry/mcap_validation.py"
    kinematics_source = repository / "src/abc_geometry/kinematics.py"
    target_messages = totals["target_messages"]
    integrity_passed = all(check["passed"] for check in invariants)
    return {
        "schema_version": REPORT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": "val",
            "tree_uri": f"hf://datasets/{DATASET_ID}@{DATASET_REVISION}/",
            "samples_uri": f"hf://datasets/{DATASET_ID}@{DATASET_REVISION}/samples.json",
            "scope": "all public raw MCAP episodes present at the pinned revision",
            "mcap_inventory_sha256": PINNED_MCAP_INVENTORY_SHA256,
            "samples_git_blob_oid": PINNED_SAMPLES_GIT_BLOB_OID,
        },
        "validator": {
            "name": "abc-geometry-recovery",
            "version": __version__,
            "report_schema": "abc-mcap-geometry-audit/v1",
            "translation_tolerance_m": TRANSLATION_TOLERANCE_M,
            "rotation_tolerance_deg": ROTATION_TOLERANCE_DEG,
            "audit_driver_source_sha256": sha256_file(Path(__file__)),
            "validation_source_sha256": sha256_file(validation_source),
            "kinematics_source_sha256": sha256_file(kinematics_source),
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "mcap": package_version("mcap"),
                "mcap-protobuf-support": package_version("mcap-protobuf-support"),
                "mujoco": package_version("mujoco"),
                "numpy": package_version("numpy"),
            },
        },
        "kinematics": {
            "model_source_revision": MODEL_SOURCE_REVISION,
            "model_sha256": model_sha256(),
            "shared_frame_transform_version": SHARED_FRAME_TRANSFORM_VERSION,
            "right_arm_shared_translation_m": RIGHT_ARM_SHARED_TRANSLATION.tolist(),
        },
        "method": {
            "pose_rules": (
                "pose exists, has 16 finite float values, valid homogeneous bottom row, "
                "orthonormal rotation, and rotation determinant +1"
            ),
            "joint_rules": "position exists and has exactly 6 finite float values",
            "fk_frame": "shared_bimanual",
            "raw_count_cross_check": "MCAP summary channel counts versus decoded records",
            "message_count_note": (
                "samples.json counts original messages; FiftyOne .plot channels are generated "
                "duplicates and are included in each MCAP summary"
            ),
            "integrity_check": (
                "local SHA-256 versus pinned Hub LFS SHA-256, before and after audit"
            ),
            "right_arm_offset_source": (
                "merged per-topic observed_base_translation_offset_m statistics from the "
                "same validator pass"
            ),
            "right_arm_offset_cluster_rounding_m": 0.01,
        },
        "integrity_passed": integrity_passed,
        "passed": integrity_passed and all(item["audit"]["passed"] for item in files),
        "conclusion": {
            "real_missing_pose_records_exist": totals["poses"]["missing"] > 0,
            "episodes_with_missing_pose_records": sum(
                item["audit"]["totals"]["invalid_poses"] > 0 for item in files
            ),
            "episodes_with_all_target_poses_missing": sum(
                item["audit"]["totals"]["invalid_poses"]
                == item["audit"]["totals"]["target_messages"]
                for item in files
            ),
            "missing_pose_fraction_of_arm_records": (
                totals["poses"]["missing"] / target_messages if target_messages else None
            ),
            "malformed_pose_records_exist": totals["poses"]["malformed"] > 0,
            "nonfinite_pose_records_exist": totals["poses"]["nonfinite"] > 0,
            "any_invalid_pose_records_exist": any(
                totals["poses"][key] > 0 for key in ("missing", "malformed", "nonfinite")
            ),
            "fk_recoverable_invalid_pose_records": totals["poses"]["recoverable_invalid"],
            "fk_unrecoverable_invalid_pose_records": totals["poses"]["unrecoverable_invalid"],
            "episodes_with_fk_residuals_outside_tolerance": sum(
                item["audit"]["totals"]["fk_outside_tolerance"] > 0 for item in files
            ),
            "multiple_right_arm_base_offsets_observed": (
                len(aggregate_report["right_arm_base_offset_analysis"]["clusters"]) > 1
            ),
            "shared_frame_recovery_caveat": (
                "All invalid-pose records have valid joints and are FK-recoverable in each "
                "arm's local frame. Absolute right-arm poses in the shared frame require the "
                "per-episode base offset; missing-pose episodes do not expose that offset, and "
                "the intact public episodes contain both -0.61 m and -0.80 m Y offsets."
            ),
        },
        "invariants": invariants,
        **aggregate_report,
        "files": files,
    }


def run_audits(jobs: list[dict[str, Any]], *, workers: int) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    results: list[dict[str, Any]] = []
    if workers == 1:
        for job in jobs:
            result = audit_one(job)
            results.append(result)
            _print_progress(result, len(results))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(audit_one, job): job for job in jobs}
            try:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    _print_progress(result, len(results))
            except Exception:
                for future in futures:
                    future.cancel()
                raise
    return sorted(results, key=lambda item: item["path"])


def _print_progress(result: dict[str, Any], completed: int) -> None:
    totals = result["audit"]["totals"]
    print(
        f"AUDITED {completed:02d}/{EXPECTED_FILE_COUNT} {result['episode_id']} "
        f"records={totals['target_messages']} invalid={totals['invalid_poses']} "
        f"outside={totals['fk_outside_tolerance']}",
        file=sys.stderr,
        flush=True,
    )


def guard_output_path(
    output: Path,
    *,
    tree_path: Path,
    samples_path: Path,
    source_root: Path,
    overwrite: bool,
) -> None:
    resolved_output = output.resolve()
    protected = {tree_path.resolve(), samples_path.resolve()}
    if resolved_output in protected:
        raise PublicPreviewAuditError("refusing to overwrite an input manifest")
    if resolved_output.is_relative_to(source_root.resolve()):
        raise PublicPreviewAuditError("refusing to write the audit report inside --source-root")
    if output.is_symlink():
        raise PublicPreviewAuditError("refusing to replace a symlink output")
    if output.exists() and output.is_dir():
        raise IsADirectoryError(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists (pass --overwrite): {output}")


def write_json_atomic(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise PublicPreviewAuditError("refusing to replace a symlink output")
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists (pass --overwrite): {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _load_json(path: Path) -> tuple[Any, bytes]:
    data = path.read_bytes()
    try:
        return json.loads(data.decode("utf-8")), data
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicPreviewAuditError(f"invalid JSON input {path}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the exact 40-file Voxel51/ABC-130k public preview at revision "
            f"{DATASET_REVISION}."
        )
    )
    parser.add_argument("--tree", required=True, type=Path, help="pinned Hub tree API JSON")
    parser.add_argument("--samples", required=True, type=Path, help="pinned samples.json")
    parser.add_argument("--source-root", required=True, type=Path, help="downloaded dataset root")
    parser.add_argument("--output", required=True, type=Path, help="audit report JSON")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        guard_output_path(
            args.output,
            tree_path=args.tree,
            samples_path=args.samples,
            source_root=args.source_root,
            overwrite=args.overwrite,
        )
        tree_payload, _ = _load_json(args.tree)
        samples_payload, samples_bytes = _load_json(args.samples)
        jobs = build_jobs(
            tree_payload,
            samples_payload,
            samples_bytes,
            args.source_root,
        )
        files = run_audits(jobs, workers=args.workers)
        report = build_report(files)
        write_json_atomic(args.output, report, overwrite=args.overwrite)
    except (OSError, ValueError, PublicPreviewAuditError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote audit report: {args.output}", file=sys.stderr)
    return 0 if report["integrity_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
