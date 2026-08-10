"""Recover arm-local end-effector poses directly from raw ABC MCAP messages.

This module emits one immutable, provenance-keyed Parquet row for every target
arm state/action message.  It never edits an MCAP and never invents a shared
bimanual transform: the only derived geometry is the YAM ``grasp_site`` pose in
the local base frame of the arm named by the source topic.

The manifest is the commit marker for the output pair.  Parquet and JSON are
both staged before publication, the Parquet is atomically renamed first, and
the manifest is atomically renamed last.  Consumers should verify the output
hash recorded by the manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from abc_geometry import __version__
from abc_geometry.kinematics import YAMKinematics
from abc_geometry.mcap_validation import ARM_TOPICS, _read_joints, _read_pose

RAW_RECOVERY_SCHEMA_VERSION = "abc-raw-pose-recovery/v1"
POSE_LAYOUT = ("x_m", "y_m", "z_m", "qw", "qx", "qy", "qz")
INVALID_POSE_VECTOR = (float("nan"),) * len(POSE_LAYOUT)

# ``geometry.status_mask`` is intentionally redundant with the string status
# columns.  It supports cheap vectorized filtering while the strings remain
# self-explanatory in ad-hoc analysis tools.
STATUS_MASK_BITS = {
    "source_pose_intact": 0,
    "joints_intact": 1,
    "fk_valid": 2,
    "recovered": 3,
}

PARQUET_COLUMNS = (
    "provenance.source_uri",
    "provenance.source_sha256",
    "provenance.task",
    "provenance.episode",
    "provenance.topic",
    "provenance.log_time_ns",
    "provenance.publish_time_ns",
    "provenance.sequence",
    "provenance.source_message_index",
    "source.arm",
    "source.stream",
    "source.pose_status",
    "source.joint_status",
    "geometry.fk_status",
    "geometry.recovery_status",
    "geometry.status_mask",
    "geometry.recovered",
    "geometry.pose.arm_local",
)


class RawRecoveryError(RuntimeError):
    """Raised when safe raw sidecar publication cannot be completed."""


@dataclass(frozen=True, slots=True)
class RawMCAPSource:
    """One local MCAP and its canonical dataset identity.

    ``canonical_uri`` should be a revision-pinned dataset URI when available,
    for example ``hf://datasets/<repo>@<commit>/data/.../episode.mcap``.  A
    local ``file://`` URI is used when it is omitted.
    """

    path: str | Path
    canonical_uri: str | None = None
    task: str | None = None
    episode: str | None = None
    expected_sha256: str | None = None
    expected_size_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class RawSourceProvenance:
    """Verified identity and recovery counts for one source MCAP."""

    uri: str
    sha256: str
    size_bytes: int
    task: str
    episode: str
    expected_sha256: str | None
    expected_size_bytes: int | None
    expected_identity_verification_status: str
    target_messages: int
    source_pose_status_counts: dict[str, int]
    joint_status_counts: dict[str, int]
    fk_status_counts: dict[str, int]
    recovery_status_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "task": self.task,
            "episode": self.episode,
            "expected_identity": {
                "sha256": self.expected_sha256,
                "size_bytes": self.expected_size_bytes,
                "verification_status": self.expected_identity_verification_status,
                "verified_fields": [
                    name
                    for name, value in (
                        ("sha256", self.expected_sha256),
                        ("size_bytes", self.expected_size_bytes),
                    )
                    if value is not None
                ],
            },
            "target_messages": self.target_messages,
            "source_pose_status_counts": self.source_pose_status_counts,
            "joint_status_counts": self.joint_status_counts,
            "fk_status_counts": self.fk_status_counts,
            "recovery_status_counts": self.recovery_status_counts,
            "rehash_verified_after_derivation": True,
        }


@dataclass(frozen=True, slots=True)
class RawRecoveryManifest:
    """JSON-safe description of a published raw-recovery sidecar."""

    created_at: str
    output_uri: str
    output_sha256: str
    manifest_uri: str
    rows: int
    source_files: tuple[RawSourceProvenance, ...]
    source_pose_status_counts: dict[str, int]
    joint_status_counts: dict[str, int]
    fk_status_counts: dict[str, int]
    recovery_status_counts: dict[str, int]
    dataset_id: str | None
    dataset_revision: str | None
    kinematics_model: str
    kinematics_model_sha256: str | None
    kinematics_source_revision: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RAW_RECOVERY_SCHEMA_VERSION,
            "created_at": self.created_at,
            "generator": {
                "name": "abc-geometry-recovery",
                "version": __version__,
            },
            "source": {
                "dataset_id": self.dataset_id,
                "revision": self.dataset_revision,
                "modified": False,
                "rehash_verified_after_derivation": True,
                "files": [source.to_dict() for source in self.source_files],
            },
            "output": {
                "uri": self.output_uri,
                "manifest_uri": self.manifest_uri,
                "sha256": self.output_sha256,
                "format": "parquet",
                "rows": self.rows,
                "columns": list(PARQUET_COLUMNS),
            },
            "geometry": {
                "frame": "arm_local",
                "site": "grasp_site",
                "pose_layout": list(POSE_LAYOUT),
                "invalid_pose_encoding": (
                    "seven NaN values; geometry.fk_status and status_mask are authoritative"
                ),
                "quaternion_convention": "Hamilton scalar-first (wxyz)",
                "shared_bimanual_transform_emitted": False,
                "kinematics_model": self.kinematics_model,
                "kinematics_model_sha256": self.kinematics_model_sha256,
                "kinematics_source_revision": self.kinematics_source_revision,
                "status_mask_bits": STATUS_MASK_BITS,
                "source_pose_status_counts": self.source_pose_status_counts,
                "joint_status_counts": self.joint_status_counts,
                "fk_status_counts": self.fk_status_counts,
                "recovery_status_counts": self.recovery_status_counts,
            },
            "publication": {
                "strategy": "staged atomic rename; manifest is commit marker",
                "verify": "sha256(output) must equal output.sha256",
            },
        }


@dataclass(frozen=True, slots=True)
class _ResolvedSource:
    path: Path
    canonical_uri: str
    task: str
    episode: str | None
    expected_sha256: str | None
    expected_size_bytes: int | None


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _nonempty(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string when supplied")
    return value.strip()


def _expected_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
    return value.lower()


def _expected_size_bytes(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("expected_size_bytes must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError("expected_size_bytes must be a non-negative integer")
    return result


def _infer_task_episode(path: Path) -> tuple[str, str]:
    parent = path.parent
    if parent.name.startswith("episode_"):
        return parent.parent.name, parent.name
    episode = path.name
    for suffix in (".fo.mcap", ".mcap"):
        if episode.endswith(suffix):
            episode = episode[: -len(suffix)]
            break
    return parent.name, episode


def _normalise_sources(
    source: str | Path | RawMCAPSource | Iterable[str | Path | RawMCAPSource],
) -> tuple[list[_ResolvedSource], tuple[Path, ...]]:
    if isinstance(source, (str, Path, RawMCAPSource)):
        raw_sources: Sequence[str | Path | RawMCAPSource] = (source,)
    else:
        raw_sources = tuple(source)
    if not raw_sources:
        raise ValueError("at least one MCAP source is required")

    candidates: list[RawMCAPSource] = []
    input_directories: list[Path] = []
    for raw in raw_sources:
        descriptor = raw if isinstance(raw, RawMCAPSource) else RawMCAPSource(raw)
        path = Path(descriptor.path).resolve()
        if path.is_dir():
            if any(
                value is not None
                for value in (
                    descriptor.canonical_uri,
                    descriptor.task,
                    descriptor.episode,
                    descriptor.expected_sha256,
                    descriptor.expected_size_bytes,
                )
            ):
                raise ValueError("directory sources cannot carry file-level identity fields")
            input_directories.append(path)
            candidates.extend(
                RawMCAPSource(candidate) for candidate in sorted(path.rglob("*.mcap"))
            )
        else:
            candidates.append(descriptor)

    resolved: list[_ResolvedSource] = []
    by_path: dict[Path, _ResolvedSource] = {}
    uris: set[str] = set()
    for descriptor in candidates:
        path = Path(descriptor.path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() != ".mcap":
            raise RawRecoveryError(f"source is not an MCAP file: {path}")
        inferred_task, _ = _infer_task_episode(path)
        uri = _nonempty(descriptor.canonical_uri, name="canonical_uri") or path.as_uri()
        item = _ResolvedSource(
            path=path,
            canonical_uri=uri,
            task=_nonempty(descriptor.task, name="task") or inferred_task,
            # Preserve whether this was explicitly supplied so session UUID
            # metadata can remain the authoritative default at decode time.
            episode=_nonempty(descriptor.episode, name="episode"),
            expected_sha256=_expected_sha256(descriptor.expected_sha256),
            expected_size_bytes=_expected_size_bytes(descriptor.expected_size_bytes),
        )
        previous = by_path.get(path)
        if previous is not None:
            if previous != item:
                raise ValueError(
                    f"source was supplied more than once with conflicting identity: {path}"
                )
            continue
        if uri in uris:
            raise ValueError(f"canonical source URI is not unique: {uri}")
        by_path[path] = item
        uris.add(uri)
        resolved.append(item)

    if not resolved:
        raise FileNotFoundError("no MCAP files found in source")
    # Canonical URIs, rather than caller ordering or local cache paths, define
    # deterministic combined-sidecar ordering.
    resolved.sort(key=lambda item: (item.canonical_uri, item.path.as_posix()))
    return resolved, tuple(dict.fromkeys(input_directories))


def _guard_paths(
    sources: Sequence[_ResolvedSource],
    input_directories: Sequence[Path],
    output: Path,
    manifest: Path,
    *,
    overwrite: bool,
) -> None:
    protected = {source.path.resolve() for source in sources}
    output_resolved = output.resolve()
    manifest_resolved = manifest.resolve()
    if output_resolved in protected or manifest_resolved in protected:
        raise RawRecoveryError("refusing to overwrite a source MCAP")
    if output_resolved == manifest_resolved:
        raise RawRecoveryError("Parquet output and manifest paths must differ")
    if output.suffix.lower() != ".parquet":
        raise RawRecoveryError("raw recovery output must use a .parquet extension")
    for directory in input_directories:
        if output_resolved.is_relative_to(directory) or manifest_resolved.is_relative_to(directory):
            raise RawRecoveryError("refusing to publish outputs inside an input directory")
    existing = [path for path in (output, manifest) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output already exists (pass overwrite=True to replace it): {names}")


def _kinematics_name(kinematics: Any) -> str:
    for attribute in ("model_id", "model_name", "name"):
        value = getattr(kinematics, attribute, None)
        if isinstance(value, str) and value:
            return value
    return type(kinematics).__name__


def _schema(metadata: dict[str, str]) -> pa.Schema:
    # A successful FK result is always exactly xyz + qwxyz.  Encoding that
    # contract physically prevents malformed pose lengths from entering a
    # sidecar through any future writer path.
    pose_type = pa.list_(pa.float64(), len(POSE_LAYOUT))
    fields = [
        pa.field("provenance.source_uri", pa.string(), nullable=False),
        pa.field("provenance.source_sha256", pa.string(), nullable=False),
        pa.field("provenance.task", pa.string(), nullable=False),
        pa.field("provenance.episode", pa.string(), nullable=False),
        pa.field("provenance.topic", pa.string(), nullable=False),
        pa.field("provenance.log_time_ns", pa.uint64(), nullable=False),
        pa.field("provenance.publish_time_ns", pa.uint64(), nullable=False),
        pa.field("provenance.sequence", pa.uint32(), nullable=False),
        pa.field("provenance.source_message_index", pa.int64(), nullable=False),
        pa.field("source.arm", pa.string(), nullable=False),
        pa.field("source.stream", pa.string(), nullable=False),
        pa.field("source.pose_status", pa.string(), nullable=False),
        pa.field("source.joint_status", pa.string(), nullable=False),
        pa.field("geometry.fk_status", pa.string(), nullable=False),
        pa.field("geometry.recovery_status", pa.string(), nullable=False),
        pa.field("geometry.status_mask", pa.uint8(), nullable=False),
        pa.field("geometry.recovered", pa.bool_(), nullable=False),
        pa.field("geometry.pose.arm_local", pose_type, nullable=False),
    ]
    return pa.schema(
        fields,
        metadata={key.encode(): value.encode() for key, value in metadata.items()},
    )


def _counter_increment(counter: dict[str, int], status: str) -> None:
    counter[status] = counter.get(status, 0) + 1


def _pose_vector(kinematics: Any, joints: np.ndarray, arm: str) -> list[float]:
    pose = kinematics.grasp_pose(joints, arm=arm, frame="arm_local")
    vector = np.asarray(pose.as_vector(), dtype=np.float64).reshape(-1)
    if vector.size != 7 or not np.isfinite(vector).all():
        raise ValueError("FK returned a non-finite pose or a vector other than length 7")
    quaternion_norm = float(np.linalg.norm(vector[3:]))
    if quaternion_norm <= np.finfo(np.float64).eps:
        raise ValueError("FK returned a zero quaternion")
    vector[3:] /= quaternion_norm
    if vector[3] < 0:
        vector[3:] *= -1
    return vector.tolist()


def _recovery_status(pose_status: str, joint_status: str, fk_status: str) -> str:
    if fk_status == "derived":
        return "source_pose_intact" if pose_status == "intact" else "recovered"
    if joint_status != "intact":
        return (
            "source_pose_intact_invalid_joints"
            if pose_status == "intact"
            else "unrecoverable_invalid_joints"
        )
    return (
        "source_pose_intact_fk_failure" if pose_status == "intact" else "unrecoverable_fk_failure"
    )


def _message_row(
    *,
    source: _ResolvedSource,
    source_sha256: str,
    episode: str,
    topic: str,
    message: Any,
    decoded: Any,
    source_message_index: int,
    kinematics: Any,
) -> dict[str, Any]:
    arm, stream = ARM_TOPICS[topic]
    pose_status, _, _ = _read_pose(decoded)
    joint_status, joints, _ = _read_joints(decoded)

    local_pose = list(INVALID_POSE_VECTOR)
    if joints is None:
        fk_status = "not_attempted_invalid_joints"
    else:
        try:
            local_pose = _pose_vector(kinematics, joints, arm)
            fk_status = "derived"
        except Exception:
            fk_status = "failure"

    recovered = pose_status != "intact" and fk_status == "derived"
    status_mask = 0
    for name, valid in (
        ("source_pose_intact", pose_status == "intact"),
        ("joints_intact", joint_status == "intact"),
        ("fk_valid", fk_status == "derived"),
        ("recovered", recovered),
    ):
        if valid:
            status_mask |= 1 << STATUS_MASK_BITS[name]

    log_time = getattr(message, "log_time", None)
    publish_time = getattr(message, "publish_time", None)
    sequence = getattr(message, "sequence", None)
    for name, value, maximum in (
        ("log_time", log_time, (1 << 64) - 1),
        ("publish_time", publish_time, (1 << 64) - 1),
        ("sequence", sequence, (1 << 32) - 1),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise RawRecoveryError(f"MCAP message {name} is not an integer")
        if int(value) < 0 or int(value) > maximum:
            raise RawRecoveryError(f"MCAP message {name} is out of range")

    return {
        "provenance.source_uri": source.canonical_uri,
        "provenance.source_sha256": source_sha256,
        "provenance.task": source.task,
        "provenance.episode": episode,
        "provenance.topic": topic,
        "provenance.log_time_ns": int(log_time),
        "provenance.publish_time_ns": int(publish_time),
        "provenance.sequence": int(sequence),
        "provenance.source_message_index": source_message_index,
        "source.arm": arm,
        "source.stream": stream,
        "source.pose_status": pose_status,
        "source.joint_status": joint_status,
        "geometry.fk_status": fk_status,
        "geometry.recovery_status": _recovery_status(pose_status, joint_status, fk_status),
        "geometry.status_mask": status_mask,
        "geometry.recovered": recovered,
        "geometry.pose.arm_local": local_pose,
    }


def _episode_from_metadata(reader: Any, fallback: str) -> str:
    session_ids: set[str] = set()
    for metadata in reader.iter_metadata():
        if getattr(metadata, "name", None) != "session-metadata":
            continue
        values = getattr(metadata, "metadata", {})
        session_id = values.get("session-uuid") if hasattr(values, "get") else None
        if isinstance(session_id, str) and session_id.strip():
            session_ids.add(session_id.strip())
    if len(session_ids) > 1:
        raise RawRecoveryError("MCAP contains conflicting session UUID metadata")
    if not session_ids:
        return fallback
    session_id = next(iter(session_ids))
    return session_id if session_id.startswith("episode_") else f"episode_{session_id}"


def _stage_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def generate_raw_pose_sidecar(
    source: str | Path | RawMCAPSource | Iterable[str | Path | RawMCAPSource],
    output_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    kinematics: YAMKinematics | Any | None = None,
    dataset_id: str | None = None,
    dataset_revision: str | None = None,
    output_uri: str | None = None,
    manifest_uri: str | None = None,
    created_at: str | None = None,
    row_group_size: int = 8192,
    overwrite: bool = False,
    require_recovery_only: bool = False,
    reader_factory: Any | None = None,
    decoder_factory: Any | None = None,
) -> RawRecoveryManifest:
    """Write a combined raw-message FK sidecar and its commit manifest.

    Inputs may be files, directories, :class:`RawMCAPSource` descriptors, or an
    iterable mixing those forms.  Directory inputs are searched recursively for
    ``*.mcap`` and outputs are forbidden inside them.  Every source is SHA-256
    hashed before decoding and rehashed after derivation; any identity or hash
    change aborts publication and removes staged artifacts.

    When ``require_recovery_only`` is true, every target row must represent a
    missing/invalid source pose successfully recovered by FK. This release gate
    is checked while both outputs are still staged.
    """

    if row_group_size <= 0:
        raise ValueError("row_group_size must be positive")
    dataset_id = _nonempty(dataset_id, name="dataset_id")
    dataset_revision = _nonempty(dataset_revision, name="dataset_revision")
    output_uri = _nonempty(output_uri, name="output_uri")
    manifest_uri = _nonempty(manifest_uri, name="manifest_uri")
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    if not isinstance(created_at, str) or not created_at.strip():
        raise ValueError("created_at must be a non-empty string")

    sources, input_directories = _normalise_sources(source)
    output = Path(output_path)
    manifest_output = (
        Path(manifest_path)
        if manifest_path is not None
        else output.with_name(f"{output.stem}.manifest.json")
    )
    _guard_paths(sources, input_directories, output, manifest_output, overwrite=overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)

    if reader_factory is None:
        from mcap.reader import make_reader

        reader_factory = make_reader
    if decoder_factory is None:
        from mcap_protobuf.decoder import DecoderFactory

        decoder_factory = DecoderFactory

    fk = kinematics if kinematics is not None else YAMKinematics()
    kinematics_model = _kinematics_name(fk)
    kinematics_model_sha256 = getattr(fk, "model_digest", None)
    kinematics_source_revision = getattr(fk, "model_source_revision", None)
    source_identities: dict[Path, tuple[int, int, int, int]] = {}
    source_hashes: dict[Path, str] = {}
    for item in sources:
        identity_before = _stat_identity(item.path)
        digest = _sha256(item.path)
        identity_after = _stat_identity(item.path)
        if identity_before != identity_after:
            raise RawRecoveryError(f"source changed during initial hashing: {item.path}")
        observed_size = identity_after[2]
        if item.expected_size_bytes is not None and observed_size != item.expected_size_bytes:
            raise RawRecoveryError(
                f"expected size mismatch for {item.path}: "
                f"expected {item.expected_size_bytes}, observed {observed_size}"
            )
        if item.expected_sha256 is not None and digest != item.expected_sha256:
            raise RawRecoveryError(
                f"expected SHA-256 mismatch for {item.path}: "
                f"expected {item.expected_sha256}, observed {digest}"
            )
        source_identities[item.path] = identity_after
        source_hashes[item.path] = digest

    parquet_metadata = {
        "abc_geometry.schema_version": RAW_RECOVERY_SCHEMA_VERSION,
        "abc_geometry.generator": f"abc-geometry-recovery/{__version__}",
        "abc_geometry.source_dataset": dataset_id or "",
        "abc_geometry.source_revision": dataset_revision or "",
        "abc_geometry.source_modified": "false",
        "abc_geometry.source_rehash_required": "true",
        "abc_geometry.frame": "arm_local",
        "abc_geometry.site": "grasp_site",
        "abc_geometry.pose_layout": json.dumps(POSE_LAYOUT),
        "abc_geometry.invalid_pose_encoding": (
            "seven NaN values; geometry.fk_status and status_mask are authoritative"
        ),
        "abc_geometry.quaternion_convention": "Hamilton scalar-first (wxyz)",
        "abc_geometry.shared_bimanual_transform_emitted": "false",
        "abc_geometry.status_mask_bits": json.dumps(STATUS_MASK_BITS, sort_keys=True),
        "abc_geometry.kinematics_model": kinematics_model,
        "abc_geometry.kinematics_model_sha256": kinematics_model_sha256 or "",
        "abc_geometry.kinematics_source_revision": kinematics_source_revision or "",
    }
    output_schema = _schema(parquet_metadata)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp.parquet", dir=output.parent
    )
    os.close(descriptor)
    staged_parquet = Path(temporary_name)
    staged_manifest: Path | None = None
    writer: pq.ParquetWriter | None = None
    source_manifests: list[RawSourceProvenance] = []
    aggregate_pose_counts: dict[str, int] = {}
    aggregate_joint_counts: dict[str, int] = {}
    aggregate_fk_counts: dict[str, int] = {}
    aggregate_recovery_counts: dict[str, int] = {}
    total_rows = 0

    try:
        writer = pq.ParquetWriter(
            staged_parquet,
            output_schema,
            compression="zstd",
            use_dictionary=[
                "provenance.source_uri",
                "provenance.source_sha256",
                "provenance.task",
                "provenance.episode",
                "provenance.topic",
                "source.arm",
                "source.stream",
                "source.pose_status",
                "source.joint_status",
                "geometry.fk_status",
                "geometry.recovery_status",
            ],
        )
        for item in sources:
            pose_counts: dict[str, int] = {}
            joint_counts: dict[str, int] = {}
            fk_counts: dict[str, int] = {}
            recovery_counts: dict[str, int] = {}
            rows: list[dict[str, Any]] = []
            source_rows = 0

            with item.path.open("rb") as stream:
                reader = reader_factory(stream, decoder_factories=[decoder_factory()])
                episode = item.episode or _episode_from_metadata(
                    reader, _infer_task_episode(item.path)[1]
                )
                records = reader.iter_decoded_messages(topics=list(ARM_TOPICS))
                for _, channel, message, decoded in records:
                    topic = str(channel.topic)
                    if topic not in ARM_TOPICS:
                        # Defensive: a conforming reader honors the topic filter.
                        continue
                    row = _message_row(
                        source=item,
                        source_sha256=source_hashes[item.path],
                        episode=episode,
                        topic=topic,
                        message=message,
                        decoded=decoded,
                        source_message_index=source_rows,
                        kinematics=fk,
                    )
                    rows.append(row)
                    source_rows += 1
                    total_rows += 1
                    for counter, status in (
                        (pose_counts, row["source.pose_status"]),
                        (joint_counts, row["source.joint_status"]),
                        (fk_counts, row["geometry.fk_status"]),
                        (recovery_counts, row["geometry.recovery_status"]),
                        (aggregate_pose_counts, row["source.pose_status"]),
                        (aggregate_joint_counts, row["source.joint_status"]),
                        (aggregate_fk_counts, row["geometry.fk_status"]),
                        (aggregate_recovery_counts, row["geometry.recovery_status"]),
                    ):
                        _counter_increment(counter, status)
                    if len(rows) >= row_group_size:
                        writer.write_table(pa.Table.from_pylist(rows, schema=output_schema))
                        rows.clear()
                if rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=output_schema))

            if require_recovery_only and recovery_counts != {"recovered": source_rows}:
                raise RawRecoveryError(
                    "require_recovery_only rejected source rows: "
                    f"{item.canonical_uri} has {recovery_counts}"
                )

            source_manifests.append(
                RawSourceProvenance(
                    uri=item.canonical_uri,
                    sha256=source_hashes[item.path],
                    size_bytes=source_identities[item.path][2],
                    task=item.task,
                    episode=episode,
                    expected_sha256=item.expected_sha256,
                    expected_size_bytes=item.expected_size_bytes,
                    expected_identity_verification_status=(
                        "verified"
                        if item.expected_sha256 is not None or item.expected_size_bytes is not None
                        else "not_supplied"
                    ),
                    target_messages=source_rows,
                    source_pose_status_counts=pose_counts,
                    joint_status_counts=joint_counts,
                    fk_status_counts=fk_counts,
                    recovery_status_counts=recovery_counts,
                )
            )

        writer.close()
        writer = None

        staged_file = pq.ParquetFile(staged_parquet)
        if staged_file.metadata.num_rows != total_rows:
            staged_rows = staged_file.metadata.num_rows
            raise RawRecoveryError(
                f"staged Parquet row count mismatch: {staged_rows} != {total_rows}"
            )
        if staged_file.schema_arrow != output_schema:
            raise RawRecoveryError("staged Parquet schema does not match the recovery contract")
        del staged_file

        # Rehash every MCAP after all decoding/FK work and before publishing.
        for item in sources:
            if _stat_identity(item.path) != source_identities[item.path]:
                raise RawRecoveryError(f"source changed while it was being processed: {item.path}")
            if _sha256(item.path) != source_hashes[item.path]:
                raise RawRecoveryError(
                    f"source content changed while it was being processed: {item.path}"
                )
            if _stat_identity(item.path) != source_identities[item.path]:
                raise RawRecoveryError(
                    f"source changed during final provenance verification: {item.path}"
                )

        output_digest = _sha256(staged_parquet)
        manifest = RawRecoveryManifest(
            created_at=created_at.strip(),
            output_uri=output_uri or output.resolve().as_uri(),
            output_sha256=output_digest,
            manifest_uri=manifest_uri or manifest_output.resolve().as_uri(),
            rows=total_rows,
            source_files=tuple(source_manifests),
            source_pose_status_counts=aggregate_pose_counts,
            joint_status_counts=aggregate_joint_counts,
            fk_status_counts=aggregate_fk_counts,
            recovery_status_counts=aggregate_recovery_counts,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            kinematics_model=kinematics_model,
            kinematics_model_sha256=kinematics_model_sha256,
            kinematics_source_revision=kinematics_source_revision,
        )
        staged_manifest = _stage_json(manifest_output, manifest.to_dict())

        # Hashing is the content proof; this last inexpensive identity check
        # narrows the remaining race between verification and publication.
        for item in sources:
            if _stat_identity(item.path) != source_identities[item.path]:
                raise RawRecoveryError(f"source changed before publication: {item.path}")

        existing = [path for path in (output, manifest_output) if path.exists()]
        if existing and not overwrite:
            names = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"output already exists: {names}")
        os.replace(staged_parquet, output)
        os.replace(staged_manifest, manifest_output)
        staged_manifest = None
        return manifest
    except Exception:
        if writer is not None:
            writer.close()
        staged_parquet.unlink(missing_ok=True)
        if staged_manifest is not None:
            staged_manifest.unlink(missing_ok=True)
        raise


write_raw_pose_sidecar = generate_raw_pose_sidecar


__all__ = [
    "INVALID_POSE_VECTOR",
    "PARQUET_COLUMNS",
    "POSE_LAYOUT",
    "RAW_RECOVERY_SCHEMA_VERSION",
    "STATUS_MASK_BITS",
    "RawMCAPSource",
    "RawRecoveryError",
    "RawRecoveryManifest",
    "RawSourceProvenance",
    "generate_raw_pose_sidecar",
    "write_raw_pose_sidecar",
]
