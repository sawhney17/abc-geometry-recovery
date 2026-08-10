#!/usr/bin/env python3
"""Build the published missing-pose sidecar from a full public audit report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from abc_geometry.raw_recovery import RawMCAPSource, RawRecoveryError, generate_raw_pose_sidecar

AUDIT_SCHEMA = "abc-130k-public-mcap-geometry-audit/v1"


def select_fully_missing_sources(
    report: dict[str, Any], source_root: Path
) -> tuple[list[RawMCAPSource], int]:
    """Select episodes whose every target arm pose is missing and recoverable."""

    if report.get("schema_version") != AUDIT_SCHEMA:
        raise ValueError(f"unsupported audit schema: {report.get('schema_version')!r}")

    root = source_root.resolve()
    selected: list[RawMCAPSource] = []
    expected_rows = 0
    for item in report.get("files", []):
        totals = item["audit"]["totals"]
        target_messages = int(totals["target_messages"])
        invalid_poses = int(totals["invalid_poses"])
        if invalid_poses == 0:
            continue
        if invalid_poses != target_messages:
            raise RawRecoveryError(
                f"refusing mixed intact/invalid episode in release slice: {item['path']}"
            )
        if int(totals["recoverable_invalid_poses"]) != invalid_poses:
            raise RawRecoveryError(f"not every invalid pose is recoverable: {item['path']}")
        if int(totals["unrecoverable_invalid_poses"]) != 0:
            raise RawRecoveryError(f"episode has unrecoverable poses: {item['path']}")

        pose_counts = {
            status: sum(int(topic["poses"][status]) for topic in item["audit"]["topics"].values())
            for status in ("missing", "malformed", "nonfinite", "intact")
        }
        if pose_counts != {
            "missing": target_messages,
            "malformed": 0,
            "nonfinite": 0,
            "intact": 0,
        }:
            raise RawRecoveryError(
                f"release slice requires literal missing poses only: {item['path']}"
            )

        relative = Path(item["path"])
        source_path = (root / relative).resolve()
        if not source_path.is_relative_to(root):
            raise RawRecoveryError(f"source path escapes --source-root: {relative}")
        selected.append(
            RawMCAPSource(
                path=source_path,
                canonical_uri=item["hf_uri"],
                task=item["task"],
                episode=item["episode_id"],
                expected_sha256=item["object"]["lfs_sha256"],
                expected_size_bytes=int(item["object"]["expected_size_bytes"]),
            )
        )
        expected_rows += target_messages

    if not selected:
        raise RawRecoveryError("audit report contains no fully missing, recoverable episodes")
    return selected, expected_rows


def _outside_source_root(path: Path, source_root: Path, *, label: str) -> None:
    if path.resolve().is_relative_to(source_root.resolve()):
        raise RawRecoveryError(f"refusing to write {label} inside --source-root")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--row-group-size", type=int, default=8192)
    parser.add_argument("--created-at")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report_path = args.audit_report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sources, expected_rows = select_fully_missing_sources(report, args.source_root)
    _outside_source_root(args.output, args.source_root, label="output")
    _outside_source_root(args.manifest, args.source_root, label="manifest")
    if args.output.resolve() == report_path or args.manifest.resolve() == report_path:
        raise RawRecoveryError("refusing to overwrite the audit report")

    dataset = report["dataset"]
    manifest = generate_raw_pose_sidecar(
        sources,
        args.output,
        manifest_path=args.manifest,
        dataset_id=dataset["id"],
        dataset_revision=dataset["revision"],
        output_uri=args.output_uri,
        manifest_uri=args.manifest_uri,
        created_at=args.created_at,
        row_group_size=args.row_group_size,
        overwrite=args.overwrite,
        require_recovery_only=True,
    )
    if manifest.rows != expected_rows:
        raise RawRecoveryError(
            f"published recovery row count mismatch: {manifest.rows} != {expected_rows}"
        )
    expected_counts = {"recovered": expected_rows}
    if manifest.recovery_status_counts != expected_counts:
        raise RawRecoveryError(
            "published recovery contains a row not classified as recovered: "
            f"{manifest.recovery_status_counts}"
        )
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
