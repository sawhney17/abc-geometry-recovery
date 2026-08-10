"""Audit ABC-130K arm geometry directly from raw MCAP episodes.

ABC ``RobotState``/``RobotAction`` protobuf messages expose six joint positions
and, when present, a row-major flattened 4x4 end-effector transform.  This
module deliberately reads only the four arm state/action topics.  Camera and
gripper streams are independent clocks and are outside the scope of this
geometry audit.

The public entry point :func:`audit_mcap` is a thin MCAP adapter around
:func:`audit_decoded_messages`.  Keeping the latter independent of MCAP makes
the validation rules easy to unit-test and lets callers feed records decoded
by an existing ingestion pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from abc_geometry.kinematics import YAMKinematics

REPORT_SCHEMA_VERSION = "abc-mcap-geometry-audit/v1"

# Topic names are part of the published ABC YAM data contract.
ARM_TOPICS: dict[str, tuple[str, str]] = {
    "/left-arm-state": ("left", "state"),
    "/right-arm-state": ("right", "state"),
    "/left-arm-action": ("left", "action"),
    "/right-arm-action": ("right", "action"),
}


@dataclass(frozen=True, slots=True)
class DecodedArmRecord:
    """A decoded protobuf record reduced to fields needed by the audit."""

    topic: str
    log_time_ns: int | None
    decoded: Any


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One sampled problem. Counts in the report are never sample-limited."""

    topic: str
    log_time_ns: int | None
    code: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "log_time_ns": self.log_time_ns,
            "code": self.code,
            "detail": self.detail,
        }


@dataclass(slots=True)
class TopicAudit:
    """Validation counters and FK residuals for one arm topic."""

    topic: str
    arm: str
    stream: str
    messages: int = 0
    missing_pose: int = 0
    malformed_pose: int = 0
    nonfinite_pose: int = 0
    intact_pose: int = 0
    recoverable_invalid_pose: int = 0
    unrecoverable_invalid_pose: int = 0
    missing_joints: int = 0
    malformed_joints: int = 0
    nonfinite_joints: int = 0
    fk_failures: int = 0
    fk_compared: int = 0
    fk_outside_tolerance: int = 0
    translation_error_sum_m: float = 0.0
    translation_error_sq_sum_m2: float = 0.0
    translation_error_max_m: float = 0.0
    rotation_error_sum_deg: float = 0.0
    rotation_error_sq_sum_deg2: float = 0.0
    rotation_error_max_deg: float = 0.0
    base_translation_offset_count: int = 0
    base_translation_offset_mean_m: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    base_translation_offset_m2_m2: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    base_translation_offset_min_m: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.full(3, np.inf, dtype=np.float64)
    )
    base_translation_offset_max_m: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.full(3, -np.inf, dtype=np.float64)
    )

    @property
    def invalid_pose(self) -> int:
        return self.missing_pose + self.malformed_pose + self.nonfinite_pose

    @property
    def invalid_joints(self) -> int:
        return self.missing_joints + self.malformed_joints + self.nonfinite_joints

    def record_residual(self, translation_m: float, rotation_deg: float, outside: bool) -> None:
        self.fk_compared += 1
        self.translation_error_sum_m += translation_m
        self.translation_error_sq_sum_m2 += translation_m * translation_m
        self.translation_error_max_m = max(self.translation_error_max_m, translation_m)
        self.rotation_error_sum_deg += rotation_deg
        self.rotation_error_sq_sum_deg2 += rotation_deg * rotation_deg
        self.rotation_error_max_deg = max(self.rotation_error_max_deg, rotation_deg)
        if outside:
            self.fk_outside_tolerance += 1

    def record_base_translation_offset(self, offset_m: npt.NDArray[np.float64]) -> None:
        """Record ``recorded_xyz - arm_local_fk_xyz`` for one intact pose."""

        self.base_translation_offset_count += 1
        delta = offset_m - self.base_translation_offset_mean_m
        self.base_translation_offset_mean_m += delta / self.base_translation_offset_count
        delta_after_mean = offset_m - self.base_translation_offset_mean_m
        self.base_translation_offset_m2_m2 += delta * delta_after_mean
        self.base_translation_offset_min_m = np.minimum(
            self.base_translation_offset_min_m, offset_m
        )
        self.base_translation_offset_max_m = np.maximum(
            self.base_translation_offset_max_m, offset_m
        )

    def to_dict(self) -> dict[str, Any]:
        compared = self.fk_compared
        translation_mean = self.translation_error_sum_m / compared if compared else None
        rotation_mean = self.rotation_error_sum_deg / compared if compared else None
        translation_rmse = (
            float(np.sqrt(self.translation_error_sq_sum_m2 / compared)) if compared else None
        )
        rotation_rmse = (
            float(np.sqrt(self.rotation_error_sq_sum_deg2 / compared)) if compared else None
        )
        offset_count = self.base_translation_offset_count
        if offset_count:
            offset_mean = self.base_translation_offset_mean_m
            offset_variance = np.maximum(self.base_translation_offset_m2_m2 / offset_count, 0.0)
            offset_std = np.sqrt(offset_variance)
            offset_summary = {
                "count": offset_count,
                "mean_xyz": offset_mean.tolist(),
                "std_xyz": offset_std.tolist(),
                "min_xyz": self.base_translation_offset_min_m.tolist(),
                "max_xyz": self.base_translation_offset_max_m.tolist(),
            }
        else:
            offset_summary = {
                "count": 0,
                "mean_xyz": None,
                "std_xyz": None,
                "min_xyz": None,
                "max_xyz": None,
            }
        return {
            "arm": self.arm,
            "stream": self.stream,
            "messages": self.messages,
            "poses": {
                "missing": self.missing_pose,
                "malformed": self.malformed_pose,
                "nonfinite": self.nonfinite_pose,
                "intact": self.intact_pose,
                "recoverable_invalid": self.recoverable_invalid_pose,
                "unrecoverable_invalid": self.unrecoverable_invalid_pose,
            },
            "joints": {
                "missing": self.missing_joints,
                "malformed": self.malformed_joints,
                "nonfinite": self.nonfinite_joints,
            },
            "fk": {
                "compared": compared,
                "failures": self.fk_failures,
                "outside_tolerance": self.fk_outside_tolerance,
                "translation_error_m": {
                    "mean": translation_mean,
                    "rmse": translation_rmse,
                    "max": self.translation_error_max_m if compared else None,
                },
                "rotation_error_deg": {
                    "mean": rotation_mean,
                    "rmse": rotation_rmse,
                    "max": self.rotation_error_max_deg if compared else None,
                },
                "observed_base_translation_offset_m": offset_summary,
            },
        }


@dataclass(slots=True)
class MCAPAuditReport:
    """Structured, JSON-safe evidence from auditing one MCAP episode."""

    source: str
    translation_tolerance_m: float
    rotation_tolerance_deg: float
    topics: dict[str, TopicAudit]
    issues: list[ValidationIssue] = field(default_factory=list)
    issue_sample_limit: int = 100
    issue_count: int = 0
    decode_errors: int = 0
    ignored_messages: int = 0

    @property
    def missing_topics(self) -> list[str]:
        return [topic for topic in ARM_TOPICS if self.topics[topic].messages == 0]

    @property
    def messages(self) -> int:
        return sum(topic.messages for topic in self.topics.values())

    @property
    def invalid_poses(self) -> int:
        return sum(topic.invalid_pose for topic in self.topics.values())

    @property
    def intact_poses(self) -> int:
        return sum(topic.intact_pose for topic in self.topics.values())

    @property
    def recoverable_invalid_poses(self) -> int:
        return sum(topic.recoverable_invalid_pose for topic in self.topics.values())

    @property
    def unrecoverable_invalid_poses(self) -> int:
        return sum(topic.unrecoverable_invalid_pose for topic in self.topics.values())

    @property
    def fk_compared(self) -> int:
        return sum(topic.fk_compared for topic in self.topics.values())

    @property
    def fk_outside_tolerance(self) -> int:
        return sum(topic.fk_outside_tolerance for topic in self.topics.values())

    @property
    def passed(self) -> bool:
        return not (
            self.missing_topics
            or self.invalid_poses
            or self.decode_errors
            or any(topic.invalid_joints or topic.fk_failures for topic in self.topics.values())
            or self.fk_outside_tolerance
        )

    def add_issue(self, issue: ValidationIssue) -> None:
        self.issue_count += 1
        if len(self.issues) < self.issue_sample_limit:
            self.issues.append(issue)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "source": self.source,
            "passed": self.passed,
            "thresholds": {
                "translation_error_m": self.translation_tolerance_m,
                "rotation_error_deg": self.rotation_tolerance_deg,
            },
            "totals": {
                "target_messages": self.messages,
                "ignored_messages": self.ignored_messages,
                "invalid_poses": self.invalid_poses,
                "intact_poses": self.intact_poses,
                "recoverable_invalid_poses": self.recoverable_invalid_poses,
                "unrecoverable_invalid_poses": self.unrecoverable_invalid_poses,
                "fk_compared": self.fk_compared,
                "fk_outside_tolerance": self.fk_outside_tolerance,
                "decode_errors": self.decode_errors,
            },
            "missing_topics": self.missing_topics,
            "topics": {topic: audit.to_dict() for topic, audit in self.topics.items()},
            "issue_count": self.issue_count,
            "issue_sample": [issue.to_dict() for issue in self.issues],
            "issue_sample_truncated": self.issue_count > len(self.issues),
        }


def _new_report(
    source: str,
    translation_tolerance_m: float,
    rotation_tolerance_deg: float,
    issue_sample_limit: int,
) -> MCAPAuditReport:
    if not np.isfinite(translation_tolerance_m) or translation_tolerance_m < 0:
        raise ValueError("translation_tolerance_m must be finite and non-negative")
    if not np.isfinite(rotation_tolerance_deg) or rotation_tolerance_deg < 0:
        raise ValueError("rotation_tolerance_deg must be finite and non-negative")
    if issue_sample_limit < 0:
        raise ValueError("issue_sample_limit must be non-negative")

    topics = {
        topic: TopicAudit(topic=topic, arm=arm, stream=stream)
        for topic, (arm, stream) in ARM_TOPICS.items()
    }
    return MCAPAuditReport(
        source=source,
        translation_tolerance_m=translation_tolerance_m,
        rotation_tolerance_deg=rotation_tolerance_deg,
        topics=topics,
        issue_sample_limit=issue_sample_limit,
    )


def _field(message: Any, name: str) -> Any:
    if isinstance(message, Mapping):
        return message.get(name)
    return getattr(message, name, None)


def _read_pose(message: Any) -> tuple[str, npt.NDArray[np.float64] | None, str]:
    raw = _field(message, "pose")
    if raw is None:
        return "missing", None, "pose field is absent"
    try:
        if len(raw) == 0:  # Empty repeated protobuf fields are the common missing-data case.
            return "missing", None, "pose field is empty"
    except TypeError:
        return "malformed", None, "pose field is not a sequence"

    try:
        flat = np.asarray(raw, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        return "malformed", None, f"pose cannot be converted to floats: {exc}"
    if flat.size != 16:
        return "malformed", None, f"pose has {flat.size} values; expected 16"
    if not np.isfinite(flat).all():
        return "nonfinite", None, "pose contains NaN or infinity"

    matrix = flat.reshape(4, 4)
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-5, rtol=0.0):
        return "malformed", None, "pose has an invalid homogeneous bottom row"

    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0.0):
        return "malformed", None, "pose rotation is not orthonormal"
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4, rtol=0.0):
        return "malformed", None, "pose rotation determinant is not +1"
    return "intact", matrix, ""


def _read_joints(message: Any) -> tuple[str, npt.NDArray[np.float64] | None, str]:
    raw = _field(message, "position")
    if raw is None:
        return "missing", None, "position field is absent"
    try:
        if len(raw) == 0:
            return "missing", None, "position field is empty"
    except TypeError:
        return "malformed", None, "position field is not a sequence"

    try:
        joints = np.asarray(raw, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        return "malformed", None, f"position cannot be converted to floats: {exc}"
    if joints.size != 6:
        return "malformed", None, f"position has {joints.size} values; expected 6"
    if not np.isfinite(joints).all():
        return "nonfinite", None, "position contains NaN or infinity"
    return "intact", joints, ""


def _rotation_error_deg(
    recorded: npt.NDArray[np.float64], predicted: npt.NDArray[np.float64]
) -> float:
    relative = predicted[:3, :3].T @ recorded[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _fk_matrix(pose: Any) -> npt.NDArray[np.float64]:
    matrix = np.asarray(pose.matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("FK returned a non-finite matrix or a shape other than (4, 4)")
    return matrix


def _fk_matrices(
    fk: Any,
    joints: npt.NDArray[np.float64],
    *,
    arm: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
    """Return shared-frame FK and, when supported, arm-local FK.

    Native :class:`YAMKinematics` exposes ``grasp_pose_pair`` so both frames
    come from one MuJoCo evaluation.  External adapters that only implement
    the historical ``grasp_pose`` interface retain the old shared-frame
    residual behavior; arm-local offset statistics are omitted if that adapter
    cannot produce the local frame.
    """

    pair_method = getattr(fk, "grasp_pose_pair", None)
    if callable(pair_method):
        try:
            local_pose, shared_pose = pair_method(joints, arm=arm)
            shared_matrix = _fk_matrix(shared_pose)
        except Exception:
            # Preserve compatibility with external adapters that expose a
            # differently shaped or partially implemented pair method.
            pass
        else:
            try:
                local_matrix = _fk_matrix(local_pose)
            except Exception:
                local_matrix = None
            return shared_matrix, local_matrix

    shared_matrix = _fk_matrix(
        fk.grasp_pose(
            joints,
            arm=arm,
            frame="shared_bimanual",
        )
    )
    try:
        local_matrix = _fk_matrix(
            fk.grasp_pose(
                joints,
                arm=arm,
                frame="arm_local",
            )
        )
    except Exception:
        local_matrix = None
    return shared_matrix, local_matrix


def _normalise_record(item: Any) -> DecodedArmRecord:
    if isinstance(item, DecodedArmRecord):
        return item

    # Native ``reader.iter_decoded_messages`` tuple:
    # (schema, channel, message, decoded_message).
    if isinstance(item, tuple) and len(item) == 4:
        _, channel, message, decoded = item
        return DecodedArmRecord(
            topic=str(channel.topic),
            log_time_ns=getattr(message, "log_time", None),
            decoded=decoded,
        )

    topic = getattr(item, "topic", None)
    if topic is not None and hasattr(item, "decoded"):
        return DecodedArmRecord(
            topic=str(topic),
            log_time_ns=getattr(item, "log_time_ns", None),
            decoded=item.decoded,
        )
    raise TypeError("records must be DecodedArmRecord instances or MCAP decoded-message 4-tuples")


def _iter_with_decode_errors(records: Iterable[Any], report: MCAPAuditReport) -> Iterator[Any]:
    iterator = iter(records)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            return
        except Exception as exc:  # A corrupt MCAP decoder should still produce a useful report.
            report.decode_errors += 1
            report.add_issue(
                ValidationIssue(
                    topic="<decoder>",
                    log_time_ns=None,
                    code="decode_error",
                    detail=f"decoder stopped: {type(exc).__name__}: {exc}",
                )
            )
            return


def audit_decoded_messages(
    records: Iterable[Any],
    *,
    source: str = "<decoded-records>",
    kinematics: YAMKinematics | Any | None = None,
    translation_tolerance_m: float = 1e-5,
    rotation_tolerance_deg: float = 1e-3,
    issue_sample_limit: int = 100,
) -> MCAPAuditReport:
    """Validate decoded ABC arm messages and compare intact poses with FK.

    ``records`` accepts :class:`DecodedArmRecord` objects or the four-tuples
    yielded by ``mcap.reader.McapReader.iter_decoded_messages``.  The recorded
    4x4 matrix is interpreted in row-major order in the published shared
    bimanual frame.  FK residuals therefore use ``frame="shared_bimanual"``.
    For intact records, the report also summarizes the observed base
    translation ``recorded_xyz - arm_local_fk_xyz`` when the kinematics adapter
    can provide an arm-local pose.
    """

    report = _new_report(
        source,
        translation_tolerance_m,
        rotation_tolerance_deg,
        issue_sample_limit,
    )
    fk = kinematics if kinematics is not None else YAMKinematics()

    for raw_record in _iter_with_decode_errors(records, report):
        try:
            record = _normalise_record(raw_record)
        except Exception as exc:
            report.decode_errors += 1
            report.add_issue(
                ValidationIssue(
                    topic="<unknown>",
                    log_time_ns=None,
                    code="record_error",
                    detail=f"invalid decoded record: {type(exc).__name__}: {exc}",
                )
            )
            continue

        if record.topic not in ARM_TOPICS:
            report.ignored_messages += 1
            continue

        topic_audit = report.topics[record.topic]
        topic_audit.messages += 1

        pose_status, recorded_pose, pose_detail = _read_pose(record.decoded)
        if pose_status != "intact":
            pose_counter = f"{pose_status}_pose"
            setattr(topic_audit, pose_counter, getattr(topic_audit, pose_counter) + 1)
            report.add_issue(
                ValidationIssue(
                    topic=record.topic,
                    log_time_ns=record.log_time_ns,
                    code=f"{pose_status}_pose",
                    detail=pose_detail,
                )
            )
        else:
            topic_audit.intact_pose += 1

        joint_status, joints, joint_detail = _read_joints(record.decoded)
        if joint_status != "intact":
            joint_counter = f"{joint_status}_joints"
            setattr(topic_audit, joint_counter, getattr(topic_audit, joint_counter) + 1)
            report.add_issue(
                ValidationIssue(
                    topic=record.topic,
                    log_time_ns=record.log_time_ns,
                    code=joint_counter,
                    detail=joint_detail,
                )
            )

        if joints is None:
            if recorded_pose is None:
                topic_audit.unrecoverable_invalid_pose += 1
            continue

        try:
            predicted, arm_local_predicted = _fk_matrices(
                fk,
                joints,
                arm=topic_audit.arm,
            )
        except Exception as exc:
            topic_audit.fk_failures += 1
            if recorded_pose is None:
                topic_audit.unrecoverable_invalid_pose += 1
            report.add_issue(
                ValidationIssue(
                    topic=record.topic,
                    log_time_ns=record.log_time_ns,
                    code="fk_failure",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        if recorded_pose is None:
            topic_audit.recoverable_invalid_pose += 1
            continue

        if arm_local_predicted is not None:
            topic_audit.record_base_translation_offset(
                recorded_pose[:3, 3] - arm_local_predicted[:3, 3]
            )

        translation_error = float(np.linalg.norm(recorded_pose[:3, 3] - predicted[:3, 3]))
        rotation_error = _rotation_error_deg(recorded_pose, predicted)
        outside = (
            translation_error > translation_tolerance_m or rotation_error > rotation_tolerance_deg
        )
        topic_audit.record_residual(translation_error, rotation_error, outside)
        if outside:
            report.add_issue(
                ValidationIssue(
                    topic=record.topic,
                    log_time_ns=record.log_time_ns,
                    code="fk_residual_outside_tolerance",
                    detail=(
                        f"translation={translation_error:.9g} m, rotation={rotation_error:.9g} deg"
                    ),
                )
            )

    return report


def audit_mcap(
    path: str | Path,
    *,
    kinematics: YAMKinematics | Any | None = None,
    translation_tolerance_m: float = 1e-5,
    rotation_tolerance_deg: float = 1e-3,
    issue_sample_limit: int = 100,
    reader_factory: Any | None = None,
    decoder_factory: Any | None = None,
) -> MCAPAuditReport:
    """Decode and audit one raw ABC ``episode.mcap`` without modifying it.

    ``reader_factory`` and ``decoder_factory`` are injectable for tests and for
    applications that already pin custom MCAP decoder implementations.
    """

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    if reader_factory is None:
        from mcap.reader import make_reader

        reader_factory = make_reader
    if decoder_factory is None:
        from mcap_protobuf.decoder import DecoderFactory

        decoder_factory = DecoderFactory

    with source_path.open("rb") as stream:
        reader = reader_factory(stream, decoder_factories=[decoder_factory()])
        records = reader.iter_decoded_messages(topics=list(ARM_TOPICS))
        return audit_decoded_messages(
            records,
            source=str(source_path),
            kinematics=kinematics,
            translation_tolerance_m=translation_tolerance_m,
            rotation_tolerance_deg=rotation_tolerance_deg,
            issue_sample_limit=issue_sample_limit,
        )


# A discoverable verb for callers who think in terms of validation rather than auditing.
validate_mcap = audit_mcap


__all__ = [
    "ARM_TOPICS",
    "DecodedArmRecord",
    "MCAPAuditReport",
    "REPORT_SCHEMA_VERSION",
    "TopicAudit",
    "ValidationIssue",
    "audit_decoded_messages",
    "audit_mcap",
    "validate_mcap",
]
