"""Generate provenance-preserving FK pose sidecars for LeRobot v3 Parquet.

The source dataset is read-only.  Every output row keeps LeRobot's canonical
``episode_index``/``frame_index``/``index``/``timestamp`` key and records the
exact source file and row from which it was derived.  Pose vectors use the
documented layout ``[x_m, y_m, z_m, qw, qx, qy, qz]``.

Only arm-local poses are emitted. The public canary audit validates a shared
bimanual frame for that cohort, but the official LeRobot shard does not
identify its rig cohort. This sidecar therefore refuses to extrapolate the
empirical right-base calibration to unverified episodes.
"""

from __future__ import annotations

import hashlib
import json
import operator
import os
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

SIDECAR_SCHEMA_VERSION = "abc-geometry-sidecar/v1"
POSE_LAYOUT = ("x_m", "y_m", "z_m", "qw", "qx", "qy", "qz")
KEY_COLUMNS = ("episode_index", "frame_index", "index", "timestamp")
INPUT_VECTOR_COLUMNS = ("observation.state", "action")

POSE_COLUMNS = (
    "observation.left_arm.pose.arm_local",
    "observation.right_arm.pose.arm_local",
    "action.left_arm.pose.arm_local",
    "action.right_arm.pose.arm_local",
)

# ``geometry.valid_mask`` bits. A bit is set only when that arm-local pose was
# derived successfully from six finite joint positions.
VALID_MASK_BITS = {
    "observation.left_arm": 0,
    "observation.right_arm": 1,
    "action.left_arm": 2,
    "action.right_arm": 3,
}

SOURCE_VECTOR_LAYOUT = {
    "observation.left_arm": {
        "column": "observation.state",
        "indices": list(range(0, 6)),
        "units": "radians",
    },
    "observation.right_arm": {
        "column": "observation.state",
        "indices": list(range(7, 13)),
        "units": "radians",
    },
    "action.left_arm": {
        "column": "action",
        "indices": list(range(0, 6)),
        "units": "radians",
    },
    "action.right_arm": {
        "column": "action",
        "indices": list(range(7, 13)),
        "units": "radians",
    },
}

SOURCE_FEATURE_NAMES = {
    "observation.state": (
        "left_arm_joint_1",
        "left_arm_joint_2",
        "left_arm_joint_3",
        "left_arm_joint_4",
        "left_arm_joint_5",
        "left_arm_joint_6",
        "left_gripper",
        "right_arm_joint_1",
        "right_arm_joint_2",
        "right_arm_joint_3",
        "right_arm_joint_4",
        "right_arm_joint_5",
        "right_arm_joint_6",
        "right_gripper",
    ),
    "action": (
        "left_arm_action_1",
        "left_arm_action_2",
        "left_arm_action_3",
        "left_arm_action_4",
        "left_arm_action_5",
        "left_arm_action_6",
        "left_gripper_action",
        "right_arm_action_1",
        "right_arm_action_2",
        "right_arm_action_3",
        "right_arm_action_4",
        "right_arm_action_5",
        "right_arm_action_6",
        "right_gripper_action",
    ),
}


class SidecarError(RuntimeError):
    """Base exception for safe sidecar generation failures."""


class SidecarSchemaError(SidecarError):
    """Raised when source Parquet does not satisfy the LeRobot row contract."""


@dataclass(frozen=True, slots=True)
class SourceFileProvenance:
    uri: str
    size_bytes: int
    sha256: str | None
    rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "rows": self.rows,
        }


@dataclass(frozen=True, slots=True)
class SidecarManifest:
    created_at: str
    output_uri: str
    output_sha256: str
    rows: int
    valid_counts: dict[str, int]
    invalid_input_counts: dict[str, int]
    derivation_failure_counts: dict[str, int]
    source_files: tuple[SourceFileProvenance, ...]
    feature_contract: dict[str, Any]
    dataset_id: str | None = None
    dataset_revision: str | None = None
    kinematics_model: str = "YAMKinematics"
    kinematics_model_sha256: str | None = None
    kinematics_source_revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "created_at": self.created_at,
            "generator": {
                "name": "abc-geometry-recovery",
                "version": __version__,
            },
            "source": {
                "dataset_id": self.dataset_id,
                "revision": self.dataset_revision,
                "files": [source.to_dict() for source in self.source_files],
                "feature_contract": self.feature_contract,
                "modified": False,
            },
            "output": {
                "uri": self.output_uri,
                "sha256": self.output_sha256,
                "rows": self.rows,
                "format": "parquet",
            },
            "geometry": {
                "kinematics_model": self.kinematics_model,
                "kinematics_model_sha256": self.kinematics_model_sha256,
                "kinematics_source_revision": self.kinematics_source_revision,
                "pose_layout": list(POSE_LAYOUT),
                "quaternion_convention": "Hamilton scalar-first (wxyz)",
                "frames": {
                    "arm_local": "the selected arm's YAM base frame",
                },
                "excluded_frames": {
                    "shared_bimanual": (
                        "not emitted: right-base calibration is not cohort-verified "
                        "for the source shard"
                    ),
                },
                "valid_mask_bits": VALID_MASK_BITS,
                "valid_counts": self.valid_counts,
                "invalid_input_mask_bits": VALID_MASK_BITS,
                "invalid_input_counts": self.invalid_input_counts,
                "derivation_failure_mask_bits": VALID_MASK_BITS,
                "derivation_failure_counts": self.derivation_failure_counts,
                "source_vector_layout": SOURCE_VECTOR_LAYOUT,
                "derivation": "deterministic forward kinematics from six arm joints",
                "field_provenance": "derived_fk",
                "confidence": "deterministic when the corresponding valid_mask bit is set",
            },
        }


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_sources(
    source: str | Path | Iterable[str | Path],
) -> tuple[list[Path], tuple[Path, ...]]:
    raw_sources: Sequence[str | Path]
    if isinstance(source, (str, Path)):
        raw_sources = (source,)
    else:
        raw_sources = tuple(source)
    if not raw_sources:
        raise ValueError("at least one source Parquet path is required")

    expanded: list[Path] = []
    input_directories: list[Path] = []
    for raw in raw_sources:
        path = Path(raw).resolve()
        if path.is_dir():
            input_directories.append(path)
            # A LeRobot dataset root contains metadata Parquets under ``meta``.
            # They are not frame tables. When the conventional ``data``
            # directory exists, process only that subtree.
            search_root = path / "data" if (path / "data").is_dir() else path
            expanded.extend(sorted(search_root.rglob("*.parquet")))
        else:
            expanded.append(path)

    sources: list[Path] = []
    seen: set[Path] = set()
    for path in expanded:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not resolved.is_file():
            raise FileNotFoundError(path)
        if resolved.suffix.lower() != ".parquet":
            raise SidecarSchemaError(f"source is not a .parquet file: {path}")
        sources.append(resolved)
        seen.add(resolved)
    if not sources:
        raise FileNotFoundError("no Parquet files found in source")
    return sources, tuple(dict.fromkeys(input_directories))


def _guard_output_paths(
    sources: Sequence[Path],
    input_directories: Sequence[Path],
    protected_inputs: Sequence[Path],
    output_path: Path,
    manifest_path: Path | None,
    *,
    overwrite: bool,
) -> None:
    source_set = {path.resolve() for path in sources}
    protected_set = source_set | {path.resolve() for path in protected_inputs}
    output_resolved = output_path.resolve()
    if output_resolved in protected_set:
        raise SidecarError("refusing to overwrite a source input")
    if output_path.suffix.lower() != ".parquet":
        raise SidecarError("sidecar output must use a .parquet extension")

    for directory in input_directories:
        if output_resolved.is_relative_to(directory):
            raise SidecarError("refusing to write the sidecar inside an input directory")

    if manifest_path is not None:
        manifest_resolved = manifest_path.resolve()
        if manifest_resolved in protected_set:
            raise SidecarError("refusing to overwrite a source input with the manifest")
        if manifest_resolved == output_resolved:
            raise SidecarError("manifest and Parquet output paths must differ")
        for directory in input_directories:
            if manifest_resolved.is_relative_to(directory):
                raise SidecarError("refusing to write the manifest inside an input directory")

    existing = [path for path in (output_path, manifest_path) if path is not None and path.exists()]
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
    # Parquet cannot round-trip a null FixedSizeList with current PyArrow
    # versions (it is decoded as a zero-length list and rejected). Keep the
    # physical type nullable/variable and enforce seven elements at derivation;
    # ``POSE_LAYOUT`` in file metadata is the semantic fixed-size contract.
    pose_type = pa.list_(pa.float64())
    fields = [
        pa.field("episode_index", pa.int64(), nullable=False),
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("index", pa.int64(), nullable=False),
        pa.field("timestamp", pa.float64(), nullable=False),
        pa.field("provenance.source_uri", pa.string(), nullable=False),
        pa.field("provenance.source_row", pa.int64(), nullable=False),
        pa.field("provenance.source_sha256", pa.string(), nullable=True),
        *(pa.field(column, pose_type, nullable=True) for column in POSE_COLUMNS),
        pa.field("geometry.valid_mask", pa.uint8(), nullable=False),
        pa.field("geometry.invalid_input_mask", pa.uint8(), nullable=False),
        pa.field("geometry.derivation_failure_mask", pa.uint8(), nullable=False),
    ]
    encoded_metadata = {key.encode(): value.encode() for key, value in metadata.items()}
    return pa.schema(fields, metadata=encoded_metadata)


def _validate_source_schema(path: Path, parquet: pq.ParquetFile) -> None:
    schema = parquet.schema_arrow
    names = set(schema.names)
    missing = [column for column in (*KEY_COLUMNS, *INPUT_VECTOR_COLUMNS) if column not in names]
    if missing:
        raise SidecarSchemaError(f"{path} is missing required columns: {', '.join(missing)}")

    for column in ("episode_index", "frame_index", "index"):
        data_type = schema.field(column).type
        if not pa.types.is_integer(data_type):
            raise SidecarSchemaError(f"{path}: {column} must have an integer Arrow type")

    timestamp_type = schema.field("timestamp").type
    if not (
        pa.types.is_integer(timestamp_type)
        or pa.types.is_floating(timestamp_type)
        or pa.types.is_decimal(timestamp_type)
    ):
        raise SidecarSchemaError(f"{path}: timestamp must have a numeric Arrow type")

    for column in INPUT_VECTOR_COLUMNS:
        data_type = schema.field(column).type
        if not (
            pa.types.is_list(data_type)
            or pa.types.is_large_list(data_type)
            or pa.types.is_fixed_size_list(data_type)
        ):
            raise SidecarSchemaError(f"{path}: {column} must have an Arrow list type")
        value_type = data_type.value_type
        if not (
            pa.types.is_integer(value_type)
            or pa.types.is_floating(value_type)
            or pa.types.is_decimal(value_type)
        ):
            raise SidecarSchemaError(f"{path}: {column} list values must be numeric")


def _feature_contract(
    info_path: str | Path | None,
    *,
    info_uri: str | None,
) -> dict[str, Any]:
    expected = {column: list(names) for column, names in SOURCE_FEATURE_NAMES.items()}
    if info_path is None:
        if info_uri is not None:
            raise ValueError("source_info_uri requires source_info_path")
        return {
            "status": "assumed_abc_v3_layout",
            "expected_names": expected,
        }

    path = Path(info_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SidecarSchemaError(f"invalid LeRobot info JSON: {path}") from exc

    features = info.get("features")
    if not isinstance(features, dict):
        raise SidecarSchemaError(f"{path} has no features object")
    for column, expected_names in expected.items():
        feature = features.get(column)
        if not isinstance(feature, dict):
            raise SidecarSchemaError(f"{path} has no {column!r} feature contract")
        if feature.get("shape") != [14]:
            raise SidecarSchemaError(
                f"{path}: {column} shape is {feature.get('shape')!r}; expected [14]"
            )
        if feature.get("names") != expected_names:
            raise SidecarSchemaError(f"{path}: {column} names do not match ABC v3 layout")

    return {
        "status": "verified_info_json",
        "uri": info_uri or path.as_uri(),
        "sha256": _sha256(path),
        "codebase_version": info.get("codebase_version"),
        "expected_names": expected,
    }


def _finite_arm_joints(vector: Any, arm: str) -> np.ndarray | None:
    try:
        values = np.asarray(vector, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None
    if values.size != 14:
        return None
    joints = values[:6] if arm == "left" else values[7:13]
    return joints if np.isfinite(joints).all() else None


def _pose_vector(kinematics: Any, joints: np.ndarray, arm: str, frame: str) -> list[float]:
    pose = kinematics.grasp_pose(joints, arm=arm, frame=frame)
    return _validate_pose_vector(pose)


def _validate_pose_vector(pose: Any) -> list[float]:
    vector = np.asarray(pose.as_vector(), dtype=np.float64).reshape(-1)
    if vector.size != 7 or not np.isfinite(vector).all():
        raise ValueError("kinematics returned a non-finite pose or a vector other than length 7")
    return vector.tolist()


def _derive_arm_pose(
    kinematics: Any,
    vector: Any,
    arm: str,
) -> tuple[list[float] | None, str]:
    joints = _finite_arm_joints(vector, arm)
    if joints is None:
        return None, "invalid_input"
    try:
        local = _pose_vector(kinematics, joints, arm, "arm_local")
    except Exception:
        # A row remains key-aligned and visibly invalid. Callers can quarantine
        # it via the derivation-failure mask instead of losing alignment.
        return None, "derivation_failure"
    return local, "valid"


def _required_int(value: Any, column: str, path: Path, row: int) -> int:
    if value is None:
        raise SidecarSchemaError(f"{path} row {row}: {column} is null")
    if isinstance(value, (bool, np.bool_)):
        raise SidecarSchemaError(f"{path} row {row}: {column} is not an integer")
    try:
        return operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SidecarSchemaError(f"{path} row {row}: {column} is not an integer") from exc


def _required_float(value: Any, column: str, path: Path, row: int) -> float:
    if value is None:
        raise SidecarSchemaError(f"{path} row {row}: {column} is null")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SidecarSchemaError(f"{path} row {row}: {column} is not numeric") from exc
    if not np.isfinite(result):
        raise SidecarSchemaError(f"{path} row {row}: {column} is non-finite")
    return result


def _derive_batch(
    batch: pa.RecordBatch,
    *,
    path: Path,
    source_uri: str,
    source_sha256: str | None,
    source_row_offset: int,
    kinematics: Any,
    output_schema: pa.Schema,
    valid_counts: dict[str, int],
    invalid_input_counts: dict[str, int],
    derivation_failure_counts: dict[str, int],
) -> pa.Table:
    values = batch.to_pydict()
    output: dict[str, list[Any]] = {field.name: [] for field in output_schema}

    for batch_row in range(batch.num_rows):
        source_row = source_row_offset + batch_row
        output["episode_index"].append(
            _required_int(values["episode_index"][batch_row], "episode_index", path, source_row)
        )
        output["frame_index"].append(
            _required_int(values["frame_index"][batch_row], "frame_index", path, source_row)
        )
        output["index"].append(_required_int(values["index"][batch_row], "index", path, source_row))
        output["timestamp"].append(
            _required_float(values["timestamp"][batch_row], "timestamp", path, source_row)
        )
        output["provenance.source_uri"].append(source_uri)
        output["provenance.source_row"].append(source_row)
        output["provenance.source_sha256"].append(source_sha256)

        valid_mask = 0
        invalid_input_mask = 0
        derivation_failure_mask = 0
        for stream, input_column in (
            ("observation", "observation.state"),
            ("action", "action"),
        ):
            vector = values[input_column][batch_row]
            for arm in ("left", "right"):
                local, status = _derive_arm_pose(kinematics, vector, arm)
                output[f"{stream}.{arm}_arm.pose.arm_local"].append(local)
                mask_name = f"{stream}.{arm}_arm"
                bit = 1 << VALID_MASK_BITS[mask_name]
                if status == "valid":
                    valid_mask |= bit
                    valid_counts[mask_name] += 1
                elif status == "invalid_input":
                    invalid_input_mask |= bit
                    invalid_input_counts[mask_name] += 1
                else:
                    derivation_failure_mask |= bit
                    derivation_failure_counts[mask_name] += 1

        output["geometry.valid_mask"].append(valid_mask)
        output["geometry.invalid_input_mask"].append(invalid_input_mask)
        output["geometry.derivation_failure_mask"].append(derivation_failure_mask)

    arrays = [pa.array(output[field.name], type=field.type) for field in output_schema]
    return pa.Table.from_arrays(arrays, schema=output_schema)


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


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _verify_source_unchanged(
    path: Path,
    expected_identity: tuple[int, int, int, int],
    expected_sha256: str | None,
) -> None:
    if _stat_identity(path) != expected_identity:
        raise SidecarError(f"source changed while it was being processed: {path}")
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise SidecarError(f"source content changed while it was being processed: {path}")
    if _stat_identity(path) != expected_identity:
        raise SidecarError(f"source changed during final provenance verification: {path}")


def generate_pose_sidecar(
    source: str | Path | Iterable[str | Path],
    output_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    kinematics: YAMKinematics | Any | None = None,
    dataset_id: str | None = None,
    dataset_revision: str | None = None,
    source_uris: Iterable[str] | None = None,
    output_uri: str | None = None,
    source_info_path: str | Path | None = None,
    source_info_uri: str | None = None,
    batch_size: int = 8192,
    hash_sources: bool = True,
    overwrite: bool = False,
) -> SidecarManifest:
    """Derive arm-local 7D grasp poses into a new Parquet sidecar.

    ``source`` may be one Parquet file, an iterable of files, or a LeRobot
    directory (its ``data/**/*.parquet`` files are processed in lexical order).
    Source bytes are never changed. Existing output is replaced only when
    ``overwrite=True`` and is categorically rejected when it aliases a source.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    feature_contract = _feature_contract(source_info_path, info_uri=source_info_uri)
    sources, input_directories = _normalise_sources(source)
    if source_uris is None:
        source_identifiers = [path.as_uri() for path in sources]
    else:
        source_identifiers = list(source_uris)
        if len(source_identifiers) != len(sources):
            raise ValueError("source_uris must contain exactly one URI per source Parquet")
        if any(not isinstance(uri, str) or not uri.strip() for uri in source_identifiers):
            raise ValueError("source_uris must contain non-empty strings")
    source_uri_by_path = dict(zip(sources, source_identifiers, strict=True))
    output = Path(output_path)
    manifest_output = Path(manifest_path) if manifest_path is not None else None
    protected_inputs = [Path(source_info_path).resolve()] if source_info_path is not None else []
    _guard_output_paths(
        sources,
        input_directories,
        protected_inputs,
        output,
        manifest_output,
        overwrite=overwrite,
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    fk = kinematics if kinematics is not None else YAMKinematics()
    kinematics_model = _kinematics_name(fk)
    kinematics_model_sha256 = getattr(fk, "model_digest", None)
    kinematics_source_revision = getattr(fk, "model_source_revision", None)
    fingerprints: list[SourceFileProvenance] = []
    source_hashes: dict[Path, str | None] = {}
    source_identities: dict[Path, tuple[int, int, int, int]] = {}
    for path in sources:
        identity_before_hash = _stat_identity(path)
        parquet = pq.ParquetFile(path)
        _validate_source_schema(path, parquet)
        source_hash = _sha256(path) if hash_sources else None
        identity_after_hash = _stat_identity(path)
        if identity_after_hash != identity_before_hash:
            raise SidecarError(f"source changed during initial provenance hashing: {path}")
        source_identities[path] = identity_after_hash
        source_hashes[path] = source_hash
        fingerprints.append(
            SourceFileProvenance(
                uri=source_uri_by_path[path],
                size_bytes=identity_after_hash[2],
                sha256=source_hash,
                rows=parquet.metadata.num_rows,
            )
        )

    metadata = {
        "abc_geometry.schema_version": SIDECAR_SCHEMA_VERSION,
        "abc_geometry.generator": f"abc-geometry-recovery/{__version__}",
        "abc_geometry.pose_layout": json.dumps(POSE_LAYOUT),
        "abc_geometry.quaternion_convention": "Hamilton scalar-first (wxyz)",
        "abc_geometry.frames.arm_local": "selected arm YAM base frame",
        "abc_geometry.excluded_frames.shared_bimanual": (
            "not emitted: right-base calibration is not cohort-verified for the source shard"
        ),
        "abc_geometry.kinematics_model": kinematics_model,
        "abc_geometry.kinematics_model_sha256": kinematics_model_sha256 or "",
        "abc_geometry.kinematics_source_revision": kinematics_source_revision or "",
        "abc_geometry.source_dataset": dataset_id or "",
        "abc_geometry.source_revision": dataset_revision or "",
        "abc_geometry.source_modified": "false",
        "abc_geometry.source_feature_contract": json.dumps(feature_contract, sort_keys=True),
        "abc_geometry.valid_mask_bits": json.dumps(VALID_MASK_BITS, sort_keys=True),
        "abc_geometry.invalid_input_mask_bits": json.dumps(VALID_MASK_BITS, sort_keys=True),
        "abc_geometry.derivation_failure_mask_bits": json.dumps(VALID_MASK_BITS, sort_keys=True),
        "abc_geometry.source_vector_layout": json.dumps(SOURCE_VECTOR_LAYOUT, sort_keys=True),
        "abc_geometry.field_provenance": "derived_fk",
    }
    output_schema = _schema(metadata)
    valid_counts = dict.fromkeys(VALID_MASK_BITS, 0)
    invalid_input_counts = dict.fromkeys(VALID_MASK_BITS, 0)
    derivation_failure_counts = dict.fromkeys(VALID_MASK_BITS, 0)
    total_rows = 0

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp.parquet", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    staged_manifest: Path | None = None
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(
            temporary,
            output_schema,
            compression="zstd",
            use_dictionary=["provenance.source_uri", "provenance.source_sha256"],
        )
        for path in sources:
            parquet = pq.ParquetFile(path)
            source_row_offset = 0
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=[*KEY_COLUMNS, *INPUT_VECTOR_COLUMNS],
            ):
                derived = _derive_batch(
                    batch,
                    path=path,
                    source_uri=source_uri_by_path[path],
                    source_sha256=source_hashes[path],
                    source_row_offset=source_row_offset,
                    kinematics=fk,
                    output_schema=output_schema,
                    valid_counts=valid_counts,
                    invalid_input_counts=invalid_input_counts,
                    derivation_failure_counts=derivation_failure_counts,
                )
                writer.write_table(derived)
                source_row_offset += batch.num_rows
                total_rows += batch.num_rows
        writer.close()
        writer = None

        expected_rows = sum(fingerprint.rows for fingerprint in fingerprints)
        if total_rows != expected_rows:
            raise SidecarError(
                f"sidecar row count mismatch: wrote {total_rows}, expected {expected_rows}"
            )

        for path in sources:
            _verify_source_unchanged(path, source_identities[path], source_hashes[path])

        output_digest = _sha256(temporary)
        manifest = SidecarManifest(
            created_at=datetime.now(timezone.utc).isoformat(),
            output_uri=output_uri or output.resolve().as_uri(),
            output_sha256=output_digest,
            rows=total_rows,
            valid_counts=valid_counts,
            invalid_input_counts=invalid_input_counts,
            derivation_failure_counts=derivation_failure_counts,
            source_files=tuple(fingerprints),
            feature_contract=feature_contract,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            kinematics_model=kinematics_model,
            kinematics_model_sha256=kinematics_model_sha256,
            kinematics_source_revision=kinematics_source_revision,
        )

        # Stage both artifacts before publishing either. The manifest is the
        # commit marker and is renamed last; consumers must verify its output
        # hash. A crash between the two atomic renames is therefore detectable
        # as a missing or stale manifest, never a silently trusted pair.
        if manifest_output is not None:
            staged_manifest = _stage_json(manifest_output, manifest.to_dict())
        existing = [
            path for path in (output, manifest_output) if path is not None and path.exists()
        ]
        if existing and not overwrite:
            names = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"output already exists: {names}")
        temporary.replace(output)
        if staged_manifest is not None and manifest_output is not None:
            staged_manifest.replace(manifest_output)
            staged_manifest = None
        return manifest
    except Exception:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        if staged_manifest is not None:
            staged_manifest.unlink(missing_ok=True)
        raise


# Alias that reads naturally from CLI/application code.
write_pose_sidecar = generate_pose_sidecar


__all__ = [
    "INPUT_VECTOR_COLUMNS",
    "KEY_COLUMNS",
    "POSE_COLUMNS",
    "POSE_LAYOUT",
    "SIDECAR_SCHEMA_VERSION",
    "SOURCE_VECTOR_LAYOUT",
    "SOURCE_FEATURE_NAMES",
    "VALID_MASK_BITS",
    "SidecarError",
    "SidecarManifest",
    "SidecarSchemaError",
    "SourceFileProvenance",
    "generate_pose_sidecar",
    "write_pose_sidecar",
]
