from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from abc_geometry import raw_recovery as raw_module
from abc_geometry.raw_recovery import (
    PARQUET_COLUMNS,
    RAW_RECOVERY_SCHEMA_VERSION,
    RawMCAPSource,
    RawRecoveryError,
    generate_raw_pose_sidecar,
)


@dataclass
class FakePose:
    vector: np.ndarray

    def as_vector(self):
        return self.vector


class FakeKinematics:
    model_id = "synthetic-yam-arm-local-v1"
    model_digest = "f" * 64
    model_source_revision = "synthetic/yam@fixed"

    def __init__(self):
        self.calls = []

    def grasp_pose(self, joints, *, arm, frame):
        assert frame == "arm_local"
        self.calls.append((np.asarray(joints).copy(), arm, frame))
        vector = np.array(
            [float(joints[0]), 1.0 if arm == "right" else 0.0, float(np.sum(joints)), 1, 0, 0, 0],
            dtype=np.float64,
        )
        return FakePose(vector)


class Decoder:
    pass


class FakeReader:
    def __init__(self, records, *, session_uuid="session-123", captured=None):
        self.records = records
        self.session_uuid = session_uuid
        self.captured = captured if captured is not None else {}

    def iter_metadata(self):
        if self.session_uuid is None:
            return iter(())
        return iter(
            [
                SimpleNamespace(
                    name="session-metadata",
                    metadata={"session-uuid": self.session_uuid},
                )
            ]
        )

    def iter_decoded_messages(self, *, topics):
        self.captured["topics"] = topics
        return iter(self.records)


def _factory(records_by_path, *, session_uuid="session-123", captured=None):
    def reader_factory(stream, *, decoder_factories):
        assert isinstance(decoder_factories[0], Decoder)
        records = records_by_path[Path(stream.name).resolve()]
        return FakeReader(records, session_uuid=session_uuid, captured=captured)

    return reader_factory


def _record(
    topic,
    *,
    pose=None,
    position=None,
    log_time=100,
    publish_time=90,
    sequence=7,
):
    if pose is None:
        pose = np.eye(4).reshape(-1).tolist()
    if position is None:
        position = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    return (
        object(),
        SimpleNamespace(topic=topic),
        SimpleNamespace(
            log_time=log_time,
            publish_time=publish_time,
            sequence=sequence,
        ),
        SimpleNamespace(pose=pose, position=position),
    )


def _generate(source, output, records, **kwargs):
    source_path = Path(source.path if isinstance(source, RawMCAPSource) else source).resolve()
    return generate_raw_pose_sidecar(
        source,
        output,
        kinematics=kwargs.pop("kinematics", FakeKinematics()),
        reader_factory=_factory({source_path: records}, captured=kwargs.pop("captured", None)),
        decoder_factory=Decoder,
        created_at="2026-08-09T00:00:00+00:00",
        **kwargs,
    )


def test_missing_source_pose_is_recovered_with_complete_message_provenance(tmp_path):
    source = tmp_path / "episode.fo.mcap"
    source.write_bytes(b"immutable raw episode")
    original = source.read_bytes()
    canonical_uri = (
        "hf://datasets/Voxel51/ABC-130k@9659e8ce4b39580f48369cc31bc2e47a217c40e7/"
        "data/train/task/episode_session-123/episode.fo.mcap"
    )
    output = tmp_path / "recovered.parquet"
    captured = {}
    expected_sha256 = hashlib.sha256(original).hexdigest()
    descriptor = RawMCAPSource(
        source,
        canonical_uri=canonical_uri,
        task="fold_skirts",
        expected_sha256=expected_sha256.upper(),
        expected_size_bytes=len(original),
    )
    public_manifest_uri = "hf://datasets/example/recovery@commit/recovered.manifest.json"

    manifest = _generate(
        descriptor,
        output,
        [_record("/left-arm-state", pose=[])],
        captured=captured,
        dataset_id="Voxel51/ABC-130k",
        dataset_revision="9659e8ce4b39580f48369cc31bc2e47a217c40e7",
        manifest_uri=public_manifest_uri,
    )

    assert source.read_bytes() == original
    assert captured["topics"] == [
        "/left-arm-state",
        "/right-arm-state",
        "/left-arm-action",
        "/right-arm-action",
    ]
    rows = pq.read_table(output).to_pydict()
    assert rows["provenance.source_uri"] == [canonical_uri]
    assert rows["provenance.source_sha256"] == [hashlib.sha256(original).hexdigest()]
    assert rows["provenance.task"] == ["fold_skirts"]
    assert rows["provenance.episode"] == ["episode_session-123"]
    assert rows["provenance.topic"] == ["/left-arm-state"]
    assert rows["provenance.log_time_ns"] == [100]
    assert rows["provenance.publish_time_ns"] == [90]
    assert rows["provenance.sequence"] == [7]
    assert rows["provenance.source_message_index"] == [0]
    assert rows["source.pose_status"] == ["missing"]
    assert rows["source.joint_status"] == ["intact"]
    assert rows["geometry.fk_status"] == ["derived"]
    assert rows["geometry.recovery_status"] == ["recovered"]
    assert rows["geometry.status_mask"] == [0b1110]
    assert rows["geometry.recovered"] == [True]
    assert rows["geometry.pose.arm_local"][0] == pytest.approx([1, 0, 21, 1, 0, 0, 0])
    assert manifest.rows == 1
    assert manifest.manifest_uri == public_manifest_uri
    assert manifest.recovery_status_counts == {"recovered": 1}

    persisted = json.loads((tmp_path / "recovered.manifest.json").read_text())
    assert persisted == manifest.to_dict()
    assert persisted["source"]["modified"] is False
    assert persisted["source"]["rehash_verified_after_derivation"] is True
    assert persisted["output"]["manifest_uri"] == public_manifest_uri
    assert persisted["geometry"]["shared_bimanual_transform_emitted"] is False
    expected_identity = persisted["source"]["files"][0]["expected_identity"]
    assert expected_identity == {
        "sha256": expected_sha256,
        "size_bytes": len(original),
        "verification_status": "verified",
        "verified_fields": ["sha256", "size_bytes"],
    }
    assert hashlib.sha256(output.read_bytes()).hexdigest() == manifest.output_sha256


def test_intact_pose_is_classified_without_being_claimed_as_recovered(tmp_path):
    source = tmp_path / "intact.mcap"
    source.write_bytes(b"intact")
    output = tmp_path / "intact.parquet"
    kinematics = FakeKinematics()

    _generate(
        RawMCAPSource(source, task="task", episode="episode_intact"),
        output,
        [_record("/right-arm-action")],
        kinematics=kinematics,
    )

    rows = pq.read_table(output).to_pydict()
    assert rows["source.arm"] == ["right"]
    assert rows["source.stream"] == ["action"]
    assert rows["source.pose_status"] == ["intact"]
    assert rows["source.joint_status"] == ["intact"]
    assert rows["geometry.fk_status"] == ["derived"]
    assert rows["geometry.recovery_status"] == ["source_pose_intact"]
    assert rows["geometry.status_mask"] == [0b0111]
    assert rows["geometry.recovered"] == [False]
    assert rows["geometry.pose.arm_local"][0] == pytest.approx([1, 1, 21, 1, 0, 0, 0])
    assert len(kinematics.calls) == 1
    assert kinematics.calls[0][1:] == ("right", "arm_local")


def test_invalid_joints_are_explicit_and_never_zero_padded_into_fk(tmp_path):
    source = tmp_path / "bad-joints.mcap"
    source.write_bytes(b"bad joints")
    output = tmp_path / "bad-joints.parquet"
    kinematics = FakeKinematics()

    manifest = _generate(
        RawMCAPSource(source, task="task", episode="episode_bad"),
        output,
        [_record("/left-arm-action", pose=[], position=[0.0] * 5)],
        kinematics=kinematics,
    )

    rows = pq.read_table(output).to_pydict()
    assert rows["source.pose_status"] == ["missing"]
    assert rows["source.joint_status"] == ["malformed"]
    assert rows["geometry.fk_status"] == ["not_attempted_invalid_joints"]
    assert rows["geometry.recovery_status"] == ["unrecoverable_invalid_joints"]
    assert rows["geometry.status_mask"] == [0]
    assert rows["geometry.recovered"] == [False]
    assert np.isnan(rows["geometry.pose.arm_local"][0]).all()
    assert len(rows["geometry.pose.arm_local"][0]) == 7
    assert kinematics.calls == []
    assert manifest.joint_status_counts == {"malformed": 1}


def test_fk_failure_is_distinct_from_invalid_joint_telemetry(tmp_path):
    source = tmp_path / "fk-failure.mcap"
    source.write_bytes(b"valid joints, failed model")
    output = tmp_path / "fk-failure.parquet"

    class BrokenKinematics(FakeKinematics):
        def grasp_pose(self, joints, *, arm, frame):
            assert frame == "arm_local"
            raise RuntimeError("synthetic FK failure")

    manifest = _generate(
        RawMCAPSource(source, task="task", episode="episode_failure"),
        output,
        [_record("/right-arm-state", pose=[])],
        kinematics=BrokenKinematics(),
    )

    rows = pq.read_table(output).to_pydict()
    assert rows["source.joint_status"] == ["intact"]
    assert rows["geometry.fk_status"] == ["failure"]
    assert rows["geometry.recovery_status"] == ["unrecoverable_fk_failure"]
    assert rows["geometry.status_mask"] == [0b0010]
    assert np.isnan(rows["geometry.pose.arm_local"][0]).all()
    assert len(rows["geometry.pose.arm_local"][0]) == 7
    assert manifest.fk_status_counts == {"failure": 1}


def test_refuses_to_overwrite_sources_or_existing_outputs(tmp_path):
    source = tmp_path / "source.mcap"
    source.write_bytes(b"do not touch")
    original = source.read_bytes()
    records = [_record("/left-arm-state", pose=[])]
    factory = _factory({source.resolve(): records})

    with pytest.raises(RawRecoveryError, match="source MCAP"):
        generate_raw_pose_sidecar(
            source,
            source,
            reader_factory=factory,
            decoder_factory=Decoder,
        )
    with pytest.raises(RawRecoveryError, match="source MCAP"):
        generate_raw_pose_sidecar(
            source,
            tmp_path / "output.parquet",
            manifest_path=source,
            reader_factory=factory,
            decoder_factory=Decoder,
        )

    output = tmp_path / "existing.parquet"
    output.write_bytes(b"existing output")
    with pytest.raises(FileExistsError, match="output already exists"):
        generate_raw_pose_sidecar(
            source,
            output,
            reader_factory=factory,
            decoder_factory=Decoder,
        )

    assert source.read_bytes() == original
    assert output.read_bytes() == b"existing output"


@pytest.mark.parametrize(
    ("descriptor_kwargs", "match"),
    [
        ({"expected_sha256": "0" * 64}, "expected SHA-256 mismatch"),
        ({"expected_size_bytes": 999}, "expected size mismatch"),
    ],
)
def test_wrong_expected_source_identity_aborts_before_decode_or_staging(
    descriptor_kwargs,
    match,
    tmp_path,
):
    source = tmp_path / "identity.mcap"
    source.write_bytes(b"actual source identity")
    output = tmp_path / "identity.parquet"
    reader_called = False

    def reader_factory(*args, **kwargs):
        nonlocal reader_called
        reader_called = True
        raise AssertionError("identity mismatch must abort before decoding")

    with pytest.raises(RawRecoveryError, match=match):
        generate_raw_pose_sidecar(
            RawMCAPSource(source, **descriptor_kwargs),
            output,
            kinematics=FakeKinematics(),
            reader_factory=reader_factory,
            decoder_factory=Decoder,
        )

    assert reader_called is False
    assert not output.exists()
    assert not (tmp_path / "identity.manifest.json").exists()
    assert not [path for path in tmp_path.iterdir() if ".tmp" in path.name]


@pytest.mark.parametrize(
    ("descriptor_kwargs", "match"),
    [
        ({"expected_sha256": "abc"}, "exactly 64 hexadecimal"),
        ({"expected_sha256": "g" * 64}, "exactly 64 hexadecimal"),
        ({"expected_size_bytes": -1}, "non-negative integer"),
        ({"expected_size_bytes": True}, "non-negative integer"),
    ],
)
def test_expected_source_identity_is_strictly_validated(
    descriptor_kwargs,
    match,
    tmp_path,
):
    source = tmp_path / "invalid-identity.mcap"
    source.write_bytes(b"source")
    output = tmp_path / "invalid-identity.parquet"

    with pytest.raises(ValueError, match=match):
        _generate(RawMCAPSource(source, **descriptor_kwargs), output, [])

    assert not output.exists()
    assert not (tmp_path / "invalid-identity.manifest.json").exists()


def test_combined_output_is_deterministic_and_schema_is_self_describing(tmp_path, monkeypatch):
    first = tmp_path / "first.mcap"
    second = tmp_path / "second.mcap"
    first.write_bytes(b"first immutable source")
    second.write_bytes(b"second immutable source")
    first_uri = "hf://datasets/example/abc@commit/data/task-a/episode-a/episode.mcap"
    second_uri = "hf://datasets/example/abc@commit/data/task-b/episode-b/episode.mcap"
    # Supply reverse order; canonical URI order must define the Parquet rows.
    sources = [
        RawMCAPSource(second, second_uri, "task-b", "episode-b"),
        RawMCAPSource(first, first_uri, "task-a", "episode-a"),
    ]
    records = {
        first.resolve(): [_record("/left-arm-state", pose=[], log_time=11, sequence=1)],
        second.resolve(): [_record("/right-arm-state", pose=[], log_time=22, sequence=2)],
    }
    hash_calls: dict[Path, int] = {}
    original_sha256 = raw_module._sha256

    def counted_hash(path, *args, **kwargs):
        resolved = Path(path).resolve()
        hash_calls[resolved] = hash_calls.get(resolved, 0) + 1
        return original_sha256(Path(path), *args, **kwargs)

    monkeypatch.setattr(raw_module, "_sha256", counted_hash)
    outputs = [tmp_path / "combined-a.parquet", tmp_path / "combined-b.parquet"]
    manifests = []
    for output in outputs:
        manifests.append(
            generate_raw_pose_sidecar(
                sources,
                output,
                kinematics=FakeKinematics(),
                reader_factory=_factory(records, session_uuid=None),
                decoder_factory=Decoder,
                dataset_id="example/abc",
                dataset_revision="commit",
                created_at="2026-08-09T00:00:00+00:00",
                row_group_size=1,
            )
        )

    assert manifests[0].output_sha256 == manifests[1].output_sha256
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    table = pq.read_table(outputs[0])
    assert table.column_names == list(PARQUET_COLUMNS)
    pose_type = table.schema.field("geometry.pose.arm_local").type
    assert pa.types.is_fixed_size_list(pose_type)
    assert pose_type.list_size == 7
    rows = table.to_pydict()
    assert rows["provenance.source_uri"] == [first_uri, second_uri]
    assert rows["provenance.episode"] == ["episode-a", "episode-b"]
    assert rows["provenance.log_time_ns"] == [11, 22]
    assert rows["provenance.sequence"] == [1, 2]
    metadata = table.schema.metadata
    assert metadata[b"abc_geometry.schema_version"].decode() == RAW_RECOVERY_SCHEMA_VERSION
    assert metadata[b"abc_geometry.frame"].decode() == "arm_local"
    assert metadata[b"abc_geometry.shared_bimanual_transform_emitted"].decode() == "false"
    assert metadata[b"abc_geometry.source_revision"].decode() == "commit"
    assert json.loads(metadata[b"abc_geometry.status_mask_bits"]) == {
        "fk_valid": 2,
        "joints_intact": 1,
        "recovered": 3,
        "source_pose_intact": 0,
    }
    assert [source.uri for source in manifests[0].source_files] == [first_uri, second_uri]
    # Each source is hashed before decode and rehashed after derivation in each run.
    assert hash_calls[first.resolve()] == 4
    assert hash_calls[second.resolve()] == 4


def test_manifest_uri_defaults_locally_and_rejects_blank_values(tmp_path):
    source = tmp_path / "manifest-uri.mcap"
    source.write_bytes(b"manifest uri")
    output = tmp_path / "manifest-uri.parquet"

    manifest = _generate(source, output, [_record("/left-arm-state", pose=[])])

    assert manifest.manifest_uri == (tmp_path / "manifest-uri.manifest.json").resolve().as_uri()
    assert (
        manifest.to_dict()["source"]["files"][0]["expected_identity"]["verification_status"]
        == "not_supplied"
    )

    invalid_output = tmp_path / "invalid-manifest-uri.parquet"
    with pytest.raises(ValueError, match="manifest_uri must be a non-empty string"):
        _generate(
            source,
            invalid_output,
            [_record("/left-arm-state", pose=[])],
            manifest_uri="   ",
        )
    assert not invalid_output.exists()


def test_decode_failure_removes_all_staged_artifacts(tmp_path):
    source = tmp_path / "broken.mcap"
    source.write_bytes(b"broken decoder source")
    output = tmp_path / "broken.parquet"

    def broken_records():
        yield _record("/left-arm-state", pose=[])
        raise ValueError("synthetic decoder failure")

    with pytest.raises(ValueError, match="synthetic decoder failure"):
        _generate(source, output, broken_records())

    assert not output.exists()
    assert not (tmp_path / "broken.manifest.json").exists()
    assert not [path for path in tmp_path.iterdir() if ".tmp" in path.name]


def test_recovery_only_release_gate_rejects_before_publication(tmp_path):
    source = tmp_path / "intact-release.mcap"
    source.write_bytes(b"intact source cannot enter missing-pose release")
    output = tmp_path / "rejected.parquet"

    with pytest.raises(RawRecoveryError, match="require_recovery_only rejected"):
        _generate(
            RawMCAPSource(source, task="task", episode="episode_intact"),
            output,
            [_record("/left-arm-state")],
            require_recovery_only=True,
        )

    assert not output.exists()
    assert not (tmp_path / "rejected.manifest.json").exists()
    assert not [path for path in tmp_path.iterdir() if ".tmp" in path.name]


def test_source_rehash_mismatch_aborts_publication(tmp_path, monkeypatch):
    source = tmp_path / "rehash.mcap"
    source.write_bytes(b"stable source bytes")
    output = tmp_path / "rehash.parquet"
    original = raw_module._sha256
    source_hash_calls = 0

    def mismatched_second_source_hash(path, *args, **kwargs):
        nonlocal source_hash_calls
        if Path(path).resolve() == source.resolve():
            source_hash_calls += 1
            if source_hash_calls == 2:
                return "0" * 64
        return original(Path(path), *args, **kwargs)

    monkeypatch.setattr(raw_module, "_sha256", mismatched_second_source_hash)

    with pytest.raises(RawRecoveryError, match="source content changed"):
        _generate(source, output, [_record("/left-arm-state", pose=[])])

    assert source_hash_calls == 2
    assert not output.exists()
    assert not (tmp_path / "rehash.manifest.json").exists()
    assert not [path for path in tmp_path.iterdir() if ".tmp" in path.name]
