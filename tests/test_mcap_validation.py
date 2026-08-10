from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from abc_geometry.mcap_validation import (
    ARM_TOPICS,
    DecodedArmRecord,
    audit_decoded_messages,
    audit_mcap,
)


@dataclass
class FakePose:
    matrix: np.ndarray


class FakeKinematics:
    def grasp_pose(self, joints, *, arm, frame):
        assert frame == "shared_bimanual"
        matrix = np.eye(4)
        matrix[:3, 3] = [float(joints[0]), -0.61 if arm == "right" else 0.0, 0.0]
        return FakePose(matrix)


class PairKinematics:
    def __init__(self):
        self.pair_calls = 0

    def grasp_pose_pair(self, joints, *, arm):
        self.pair_calls += 1
        local = np.eye(4)
        local[:3, 3] = [float(joints[0]), 0.0, 0.0]
        shared = local.copy()
        if arm == "right":
            shared[1, 3] = -0.61
        return FakePose(local), FakePose(shared)

    def grasp_pose(self, joints, *, arm, frame):
        raise AssertionError("grasp_pose_pair should avoid a second FK evaluation")


def _matrix(joint_0: float = 0.0, arm: str = "left") -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = [joint_0, -0.61 if arm == "right" else 0.0, 0.0]
    return matrix


def _message(*, arm="left", joint_0=0.0, pose=None, position=None):
    if pose is None:
        pose = _matrix(joint_0, arm).reshape(-1).tolist()
    if position is None:
        position = [joint_0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return SimpleNamespace(pose=pose, position=position)


def _complete_records():
    records = []
    for timestamp, (topic, (arm, _)) in enumerate(ARM_TOPICS.items()):
        records.append(
            DecodedArmRecord(
                topic=topic,
                log_time_ns=timestamp,
                decoded=_message(arm=arm, joint_0=timestamp / 10),
            )
        )
    return records


def test_audit_matches_all_intact_topics_against_fk():
    report = audit_decoded_messages(_complete_records(), kinematics=FakeKinematics())

    assert report.passed
    assert report.messages == 4
    assert report.intact_poses == 4
    assert report.invalid_poses == 0
    assert report.fk_compared == 4
    assert report.fk_outside_tolerance == 0
    assert report.missing_topics == []

    payload = report.to_dict()
    assert payload["schema_version"] == "abc-mcap-geometry-audit/v1"
    assert payload["totals"]["fk_compared"] == 4
    assert payload["topics"]["/right-arm-action"]["fk"]["translation_error_m"][
        "max"
    ] == pytest.approx(0.0)


def test_audit_detects_exact_observed_right_base_translation_offset():
    recorded = np.eye(4)
    recorded[:3, 3] = [0.25, -0.80, 0.0]
    kinematics = PairKinematics()
    records = [
        DecodedArmRecord(
            "/right-arm-state",
            1,
            _message(
                arm="right",
                joint_0=0.25,
                pose=recorded.reshape(-1).tolist(),
            ),
        )
    ]

    report = audit_decoded_messages(records, kinematics=kinematics)
    topic = report.topics["/right-arm-state"]
    summary = topic.to_dict()["fk"]["observed_base_translation_offset_m"]

    assert kinematics.pair_calls == 1
    assert topic.fk_compared == 1
    assert topic.translation_error_max_m == pytest.approx(0.19)
    assert topic.fk_outside_tolerance == 1
    assert summary["count"] == 1
    assert summary["mean_xyz"] == pytest.approx([0.0, -0.80, 0.0])
    assert summary["std_xyz"] == pytest.approx([0.0, 0.0, 0.0])
    assert summary["min_xyz"] == pytest.approx([0.0, -0.80, 0.0])
    assert summary["max_xyz"] == pytest.approx([0.0, -0.80, 0.0])


def test_observed_base_translation_offset_reports_no_data_explicitly():
    kinematics = PairKinematics()
    report = audit_decoded_messages([], kinematics=kinematics)

    summary = report.to_dict()["topics"]["/right-arm-state"]["fk"][
        "observed_base_translation_offset_m"
    ]

    assert kinematics.pair_calls == 0
    assert summary == {
        "count": 0,
        "mean_xyz": None,
        "std_xyz": None,
        "min_xyz": None,
        "max_xyz": None,
    }


def test_audit_classifies_pose_failures_without_conflating_them():
    records = _complete_records()
    records.extend(
        [
            DecodedArmRecord("/left-arm-state", 10, _message(pose=[], position=[0.0] * 6)),
            DecodedArmRecord("/left-arm-state", 11, _message(pose=[0.0] * 15, position=[0.0] * 6)),
            DecodedArmRecord(
                "/left-arm-state",
                12,
                _message(pose=[float("nan")] + [0.0] * 15, position=[0.0] * 6),
            ),
        ]
    )

    report = audit_decoded_messages(records, kinematics=FakeKinematics())
    topic = report.topics["/left-arm-state"]

    assert not report.passed
    assert topic.messages == 4
    assert topic.missing_pose == 1
    assert topic.malformed_pose == 1
    assert topic.nonfinite_pose == 1
    assert topic.intact_pose == 1
    assert topic.recoverable_invalid_pose == 3
    assert topic.unrecoverable_invalid_pose == 0
    assert report.invalid_poses == 3
    assert report.recoverable_invalid_poses == 3
    assert {issue.code for issue in report.issues} >= {
        "missing_pose",
        "malformed_pose",
        "nonfinite_pose",
    }


def test_audit_reports_bad_joints_and_fk_residuals():
    records = _complete_records()
    shifted = _matrix()
    shifted[0, 3] = 0.2
    records.append(
        DecodedArmRecord(
            "/left-arm-action",
            20,
            _message(pose=shifted.reshape(-1).tolist(), position=[0.0] * 6),
        )
    )
    records.append(
        DecodedArmRecord(
            "/right-arm-action",
            21,
            _message(arm="right", pose=_matrix(arm="right").reshape(-1), position=[1.0] * 5),
        )
    )

    report = audit_decoded_messages(
        records,
        kinematics=FakeKinematics(),
        translation_tolerance_m=0.01,
    )

    assert report.fk_outside_tolerance == 1
    left_action = report.topics["/left-arm-action"]
    assert left_action.fk_compared == 2
    assert left_action.translation_error_max_m == pytest.approx(0.2)
    assert left_action.fk_outside_tolerance == 1
    assert report.topics["/right-arm-action"].malformed_joints == 1
    assert {issue.code for issue in report.issues} >= {
        "malformed_joints",
        "fk_residual_outside_tolerance",
    }


def test_invalid_pose_without_valid_joints_is_not_marked_recoverable():
    records = _complete_records()
    records.append(
        DecodedArmRecord(
            "/left-arm-state",
            30,
            _message(pose=[], position=[0.0] * 5),
        )
    )

    report = audit_decoded_messages(records, kinematics=FakeKinematics())
    topic = report.topics["/left-arm-state"]

    assert topic.recoverable_invalid_pose == 0
    assert topic.unrecoverable_invalid_pose == 1
    assert report.to_dict()["totals"]["unrecoverable_invalid_poses"] == 1


def test_missing_topics_and_ignored_messages_are_explicit():
    records = [
        DecodedArmRecord("/left-arm-state", 1, _message()),
        DecodedArmRecord("/camera/top", 2, object()),
    ]
    report = audit_decoded_messages(records, kinematics=FakeKinematics())

    assert not report.passed
    assert report.ignored_messages == 1
    assert report.missing_topics == [
        "/right-arm-state",
        "/left-arm-action",
        "/right-arm-action",
    ]


def test_native_mcap_tuple_shape_is_supported():
    channel = SimpleNamespace(topic="/left-arm-state")
    envelope = SimpleNamespace(log_time=123)
    record = (object(), channel, envelope, _message())

    report = audit_decoded_messages([record], kinematics=FakeKinematics())

    assert report.topics["/left-arm-state"].messages == 1
    assert report.issues == []


def test_audit_mcap_uses_decoder_and_topic_filter(tmp_path):
    source = tmp_path / "episode.mcap"
    source.write_bytes(b"synthetic placeholder")
    captured = {}

    class Decoder:
        pass

    class Reader:
        def iter_decoded_messages(self, *, topics):
            captured["topics"] = topics
            return iter(_complete_records())

    def reader_factory(stream, *, decoder_factories):
        captured["bytes"] = stream.read()
        captured["decoder"] = decoder_factories[0]
        return Reader()

    report = audit_mcap(
        source,
        kinematics=FakeKinematics(),
        reader_factory=reader_factory,
        decoder_factory=Decoder,
    )

    assert report.passed
    assert captured["bytes"] == b"synthetic placeholder"
    assert isinstance(captured["decoder"], Decoder)
    assert captured["topics"] == list(ARM_TOPICS)


def test_decoder_failure_is_returned_as_structured_evidence():
    def broken_records():
        yield _complete_records()[0]
        raise ValueError("bad protobuf payload")

    report = audit_decoded_messages(broken_records(), kinematics=FakeKinematics())

    assert report.decode_errors == 1
    assert not report.passed
    assert report.issues[-1].code == "decode_error"


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_validation_rejects_invalid_translation_thresholds(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        audit_decoded_messages([], translation_tolerance_m=value)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_validation_rejects_invalid_rotation_thresholds(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        audit_decoded_messages([], rotation_tolerance_deg=value)
