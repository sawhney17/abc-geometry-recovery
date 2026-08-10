#!/usr/bin/env python3
"""Cross-check an ABC audit without using its pose/joint classification code."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

TOPICS = (
    "/left-arm-state",
    "/right-arm-state",
    "/left-arm-action",
    "/right-arm-action",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def scan(path: Path) -> dict[str, Any]:
    """Count literal protobuf field lengths and finiteness for four arm topics."""

    pose_lengths: Counter[int] = Counter()
    joint_lengths: Counter[int] = Counter()
    topic_counts: Counter[str] = Counter()
    nonfinite_pose = 0
    nonfinite_joints = 0
    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, channel, _, decoded in reader.iter_decoded_messages(topics=list(TOPICS)):
            if channel.topic not in TOPICS:
                continue
            pose = list(decoded.pose)
            joints = list(decoded.position)
            topic_counts[channel.topic] += 1
            pose_lengths[len(pose)] += 1
            joint_lengths[len(joints)] += 1
            nonfinite_pose += any(not math.isfinite(float(value)) for value in pose)
            nonfinite_joints += any(not math.isfinite(float(value)) for value in joints)
    return {
        "target_messages": sum(topic_counts.values()),
        "topic_counts": dict(sorted(topic_counts.items())),
        "pose_length_histogram": {str(key): value for key, value in sorted(pose_lengths.items())},
        "joint_length_histogram": {str(key): value for key, value in sorted(joint_lengths.items())},
        "nonfinite_pose_records": nonfinite_pose,
        "nonfinite_joint_records": nonfinite_joints,
    }


def build_cross_check(audit_report: Path, root: Path) -> dict[str, Any]:
    report = json.loads(audit_report.read_text(encoding="utf-8"))
    files = []
    for item in sorted(report["files"], key=lambda value: value["path"]):
        source = (root / item["path"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = sha256_file(source)
        result = scan(source)
        result.update(
            {
                "path": item["path"],
                "hf_uri": item["hf_uri"],
                "task": item["task"],
                "episode_id": item["episode_id"],
                "size_bytes": source.stat().st_size,
                "sha256": digest,
                "sha256_matches_primary_audit": digest == item["object"]["sha256_before"],
            }
        )
        files.append(result)

    pose_histogram: Counter[str] = Counter()
    joint_histogram: Counter[str] = Counter()
    for item in files:
        pose_histogram.update(item["pose_length_histogram"])
        joint_histogram.update(item["joint_length_histogram"])
    totals = {
        "files": len(files),
        "source_bytes": sum(item["size_bytes"] for item in files),
        "target_messages": sum(item["target_messages"] for item in files),
        "pose_length_histogram": dict(
            sorted(pose_histogram.items(), key=lambda item: int(item[0]))
        ),
        "joint_length_histogram": dict(
            sorted(joint_histogram.items(), key=lambda item: int(item[0]))
        ),
        "nonfinite_pose_records": sum(item["nonfinite_pose_records"] for item in files),
        "nonfinite_joint_records": sum(item["nonfinite_joint_records"] for item in files),
    }
    expected = report["totals"]
    matches = {
        "all_source_hashes": all(item["sha256_matches_primary_audit"] for item in files),
        "target_messages": totals["target_messages"] == expected["target_messages"],
        "missing_pose_records": totals["pose_length_histogram"].get("0", 0)
        == expected["poses"]["missing"],
        "intact_length_pose_records": totals["pose_length_histogram"].get("16", 0)
        == expected["poses"]["intact"],
        "only_expected_pose_lengths": set(totals["pose_length_histogram"]) <= {"0", "16"},
        "six_joint_records": totals["joint_length_histogram"] == {"6": expected["target_messages"]},
        "nonfinite_pose_records": totals["nonfinite_pose_records"]
        == expected["poses"]["nonfinite"],
        "nonfinite_joint_records": totals["nonfinite_joint_records"] == 0,
    }
    return {
        "schema_version": "abc-130k-independent-protobuf-field-scan/v1",
        "dataset": report["dataset"],
        "independent_of": "abc_geometry.mcap_validation pose/joint classification helpers",
        "passed": all(matches.values()),
        "matches_primary_audit": matches,
        "totals": totals,
        "files": files,
    }


def write_json_atomic(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    protected = {args.audit_report.resolve()}
    report = json.loads(args.audit_report.read_text(encoding="utf-8"))
    protected.update((args.root / item["path"]).resolve() for item in report["files"])
    if args.output.resolve() in protected:
        raise ValueError("refusing to overwrite an input with the cross-check report")

    payload = build_cross_check(args.audit_report, args.root)
    write_json_atomic(args.output, payload, overwrite=args.overwrite)
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
