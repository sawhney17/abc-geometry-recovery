"""Command-line interface for reproducible ABC geometry audits and recovery."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, TypeVar

from abc_geometry import __version__
from abc_geometry.mcap_validation import MCAPAuditReport, audit_mcap
from abc_geometry.raw_recovery import RawMCAPSource, generate_raw_pose_sidecar
from abc_geometry.sidecar import generate_pose_sidecar

T = TypeVar("T")


def _write_json_atomic(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"report already exists (pass --overwrite): {path}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _audit_payload(reports: list[MCAPAuditReport]) -> dict[str, Any]:
    return {
        "schema_version": "abc-mcap-geometry-audit-batch/v1",
        "generator": {"name": "abc-geometry-recovery", "version": __version__},
        "passed": all(report.passed for report in reports),
        "totals": {
            "files": len(reports),
            "target_messages": sum(report.messages for report in reports),
            "invalid_poses": sum(report.invalid_poses for report in reports),
            "recoverable_invalid_poses": sum(
                report.recoverable_invalid_poses for report in reports
            ),
            "unrecoverable_invalid_poses": sum(
                report.unrecoverable_invalid_poses for report in reports
            ),
            "fk_compared": sum(report.fk_compared for report in reports),
            "fk_outside_tolerance": sum(report.fk_outside_tolerance for report in reports),
            "decode_errors": sum(report.decode_errors for report in reports),
        },
        "files": [report.to_dict() for report in reports],
    }


def _run_audit(args: argparse.Namespace) -> int:
    if args.report is not None:
        report_path = args.report.resolve()
        input_paths = {path.resolve() for path in args.mcap}
        if report_path in input_paths:
            raise ValueError("refusing to overwrite an input MCAP with the audit report")

    reports = [
        audit_mcap(
            path,
            translation_tolerance_m=args.translation_tolerance_m,
            rotation_tolerance_deg=args.rotation_tolerance_deg,
            issue_sample_limit=args.issue_sample_limit,
        )
        for path in args.mcap
    ]
    payload = _audit_payload(reports)

    for report in reports:
        verdict = "PASS" if report.passed else "FAIL"
        print(
            f"{verdict} {report.source}: {report.messages} arm records, "
            f"{report.invalid_poses} invalid pose, "
            f"{report.recoverable_invalid_poses} FK-recoverable, "
            f"{report.fk_outside_tolerance} intact residual failures",
            file=sys.stderr,
        )

    if args.report is None:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _write_json_atomic(args.report, payload, overwrite=args.overwrite)
        print(f"wrote audit report: {args.report}", file=sys.stderr)
    return 0 if payload["passed"] else 1


def _default_manifest_path(output: Path) -> Path:
    return output.with_suffix(".manifest.json")


def _run_sidecar(args: argparse.Namespace) -> int:
    manifest_path = args.manifest or _default_manifest_path(args.output)
    if args.info_json is None:
        print(
            "warning: no --info-json supplied; the ABC 14-D feature-name layout is assumed",
            file=sys.stderr,
        )
    manifest = generate_pose_sidecar(
        args.source,
        args.output,
        manifest_path=manifest_path,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        source_uris=args.source_uri,
        output_uri=args.output_uri,
        source_info_path=args.info_json,
        source_info_uri=args.info_uri,
        batch_size=args.batch_size,
        hash_sources=not args.skip_source_hash,
        overwrite=args.overwrite,
    )
    json.dump(manifest.to_dict(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    print(
        f"wrote {manifest.rows} key-aligned rows to {args.output}; manifest: {manifest_path}",
        file=sys.stderr,
    )
    return 0


def _aligned_source_values(
    values: list[T] | None,
    *,
    option: str,
    source_count: int,
) -> list[T | None]:
    if values is None:
        return [None] * source_count
    if len(values) != source_count:
        raise ValueError(
            f"{option} must be repeated exactly once per source MCAP "
            f"({len(values)} supplied for {source_count} sources)"
        )
    return values


def _run_recover_mcap(args: argparse.Namespace) -> int:
    source_count = len(args.source)
    source_uris = _aligned_source_values(
        args.source_uri,
        option="--source-uri",
        source_count=source_count,
    )
    tasks = _aligned_source_values(
        args.task,
        option="--task",
        source_count=source_count,
    )
    episodes = _aligned_source_values(
        args.episode,
        option="--episode",
        source_count=source_count,
    )
    source_sha256s = _aligned_source_values(
        args.source_sha256,
        option="--source-sha256",
        source_count=source_count,
    )
    source_sizes = _aligned_source_values(
        args.source_size_bytes,
        option="--source-size-bytes",
        source_count=source_count,
    )
    sources = [
        RawMCAPSource(path, source_uri, task, episode, source_sha256, source_size)
        for path, source_uri, task, episode, source_sha256, source_size in zip(
            args.source,
            source_uris,
            tasks,
            episodes,
            source_sha256s,
            source_sizes,
            strict=True,
        )
    ]
    manifest_path = args.manifest or _default_manifest_path(args.output)
    manifest = generate_raw_pose_sidecar(
        sources,
        args.output,
        manifest_path=manifest_path,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        output_uri=args.output_uri,
        manifest_uri=args.manifest_uri,
        row_group_size=args.row_group_size,
        overwrite=args.overwrite,
    )
    json.dump(manifest.to_dict(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    print(
        f"recovered {manifest.rows} raw arm messages to {args.output}; manifest: {manifest_path}",
        file=sys.stderr,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="abc-geometry",
        description=(
            "Audit raw ABC-130K poses against YAM forward kinematics and emit "
            "immutable geometry sidecars for the official LeRobot conversion."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser(
        "audit-mcap", help="validate raw arm poses and identify FK-recoverable records"
    )
    audit.add_argument("mcap", nargs="+", type=Path, help="raw ABC episode.mcap files")
    audit.add_argument("--report", type=Path, help="write combined JSON instead of stdout")
    audit.add_argument("--translation-tolerance-m", type=float, default=1e-5)
    audit.add_argument("--rotation-tolerance-deg", type=float, default=1e-3)
    audit.add_argument("--issue-sample-limit", type=int, default=100)
    audit.add_argument("--overwrite", action="store_true")
    audit.set_defaults(handler=_run_audit)

    sidecar = commands.add_parser(
        "derive-sidecar",
        help="derive observed/commanded grasp poses from official 14-D ABC LeRobot rows",
    )
    sidecar.add_argument(
        "source", nargs="+", type=Path, help="source Parquet file(s) or dataset directory"
    )
    sidecar.add_argument("--output", required=True, type=Path, help="new sidecar Parquet")
    sidecar.add_argument(
        "--manifest",
        type=Path,
        help="manifest JSON (default: OUTPUT with .manifest.json suffix)",
    )
    sidecar.add_argument("--dataset-id", help="source Hub dataset ID")
    sidecar.add_argument("--dataset-revision", help="immutable source revision")
    sidecar.add_argument(
        "--source-uri",
        action="append",
        help="canonical URI for each source, in positional order (repeat per source)",
    )
    sidecar.add_argument(
        "--output-uri",
        help="canonical URI recorded for the sidecar (defaults to its local file URI)",
    )
    sidecar.add_argument(
        "--info-json",
        type=Path,
        help="LeRobot meta/info.json used to verify the 14-D feature-name contract",
    )
    sidecar.add_argument(
        "--info-uri",
        help="canonical URI recorded for --info-json (defaults to its local file URI)",
    )
    sidecar.add_argument("--batch-size", type=int, default=8192)
    sidecar.add_argument(
        "--skip-source-hash",
        action="store_true",
        help="skip source SHA-256 calculation (faster, weaker provenance)",
    )
    sidecar.add_argument("--overwrite", action="store_true")
    sidecar.set_defaults(handler=_run_sidecar)

    recover = commands.add_parser(
        "recover-mcap",
        help="derive provenance-keyed arm-local poses for raw MCAP arm messages",
    )
    recover.add_argument("source", nargs="+", type=Path, help="source ABC MCAP files")
    recover.add_argument("--output", required=True, type=Path, help="combined recovery Parquet")
    recover.add_argument(
        "--manifest",
        type=Path,
        help="local manifest JSON path (default: OUTPUT with .manifest.json suffix)",
    )
    recover.add_argument("--dataset-id", help="source dataset ID")
    recover.add_argument("--dataset-revision", help="immutable source dataset revision")
    recover.add_argument(
        "--source-uri",
        action="append",
        help="canonical URI in positional source order (repeat exactly once per source)",
    )
    recover.add_argument(
        "--task",
        action="append",
        help="canonical task in positional source order (repeat exactly once per source)",
    )
    recover.add_argument(
        "--episode",
        action="append",
        help="canonical episode in positional source order (repeat exactly once per source)",
    )
    recover.add_argument(
        "--source-sha256",
        action="append",
        help="expected SHA-256 in positional source order (repeat exactly once per source)",
    )
    recover.add_argument(
        "--source-size-bytes",
        action="append",
        type=int,
        help="expected byte size in positional source order (repeat exactly once per source)",
    )
    recover.add_argument(
        "--output-uri",
        help="public URI recorded for the Parquet (defaults to its local file URI)",
    )
    recover.add_argument(
        "--manifest-uri",
        help="public URI recorded for the manifest (defaults to its local file URI)",
    )
    recover.add_argument("--row-group-size", type=int, default=8192)
    recover.add_argument("--overwrite", action="store_true")
    recover.set_defaults(handler=_run_recover_mcap)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit status."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
