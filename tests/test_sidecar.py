from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from abc_geometry import sidecar as sidecar_module
from abc_geometry.sidecar import (
    POSE_COLUMNS,
    SIDECAR_SCHEMA_VERSION,
    SOURCE_FEATURE_NAMES,
    SidecarError,
    SidecarSchemaError,
    generate_pose_sidecar,
)


@dataclass
class FakePose:
    vector: np.ndarray

    def as_vector(self):
        return self.vector


class FakeKinematics:
    model_id = "synthetic-yam-v1"

    def grasp_pose(self, joints, *, arm, frame):
        assert frame == "arm_local"
        position = np.array(
            [
                float(joints[0]),
                0.0,
                float(np.sum(joints)),
            ]
        )
        return FakePose(np.concatenate([position, [1.0, 0.0, 0.0, 0.0]]))


def _vector(left_start: float, right_start: float):
    return [
        *np.arange(left_start, left_start + 6).tolist(),
        0.5,
        *np.arange(right_start, right_start + 6).tolist(),
        0.75,
    ]


def _write_source(path, *, invalid_right_action=False):
    state = [_vector(0, 10), _vector(1, 11)]
    action = [_vector(2, 12), _vector(3, 13)]
    if invalid_right_action:
        action[1][9] = float("nan")
    table = pa.table(
        {
            "episode_index": pa.array([4, 4], type=pa.int64()),
            "frame_index": pa.array([0, 1], type=pa.int64()),
            "index": pa.array([100, 101], type=pa.int64()),
            "timestamp": pa.array([0.0, 1 / 30], type=pa.float32()),
            "observation.state": pa.array(state, type=pa.list_(pa.float32(), 14)),
            "action": pa.array(action, type=pa.list_(pa.float32(), 14)),
            "unrelated.payload": pa.array(["preserved elsewhere", "not copied"]),
        }
    )
    pq.write_table(table, path)


def _write_info(path, *, corrupt_action_names=False):
    features = {
        column: {"shape": [14], "names": list(names)}
        for column, names in SOURCE_FEATURE_NAMES.items()
    }
    if corrupt_action_names:
        features["action"]["names"][0] = "unknown_joint"
    path.write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "features": features,
            }
        )
    )


def test_generates_keyed_pose_sidecar_and_manifest_without_touching_source(tmp_path):
    source = tmp_path / "episode_000004.parquet"
    output = tmp_path / "derived" / "geometry.parquet"
    manifest_path = tmp_path / "derived" / "geometry.manifest.json"
    _write_source(source, invalid_right_action=True)
    original_bytes = source.read_bytes()
    original_hash = hashlib.sha256(original_bytes).hexdigest()

    manifest = generate_pose_sidecar(
        source,
        output,
        manifest_path=manifest_path,
        kinematics=FakeKinematics(),
        dataset_id="lerobot/abc_130k_v3_val",
        dataset_revision="synthetic-revision",
        batch_size=1,
    )

    assert source.read_bytes() == original_bytes
    assert manifest.rows == 2
    assert manifest.source_files[0].sha256 == original_hash
    assert manifest.valid_counts == {
        "observation.left_arm": 2,
        "observation.right_arm": 2,
        "action.left_arm": 2,
        "action.right_arm": 1,
    }

    sidecar = pq.read_table(output)
    assert sidecar.num_rows == 2
    assert sidecar.column_names == [
        "episode_index",
        "frame_index",
        "index",
        "timestamp",
        "provenance.source_uri",
        "provenance.source_row",
        "provenance.source_sha256",
        *POSE_COLUMNS,
        "geometry.valid_mask",
        "geometry.invalid_input_mask",
        "geometry.derivation_failure_mask",
    ]
    rows = sidecar.to_pydict()
    assert rows["episode_index"] == [4, 4]
    assert rows["frame_index"] == [0, 1]
    assert rows["index"] == [100, 101]
    assert rows["provenance.source_row"] == [0, 1]
    assert rows["provenance.source_sha256"] == [original_hash, original_hash]
    assert rows["geometry.valid_mask"] == [0b1111, 0b0111]
    assert rows["geometry.invalid_input_mask"] == [0, 0b1000]
    assert rows["geometry.derivation_failure_mask"] == [0, 0]

    # Position then Hamilton scalar-first quaternion: xyz + qwxyz.
    assert rows["observation.left_arm.pose.arm_local"][0] == pytest.approx(
        [0.0, 0.0, 15.0, 1.0, 0.0, 0.0, 0.0]
    )
    assert rows["observation.right_arm.pose.arm_local"][0] == pytest.approx(
        [10.0, 0.0, 75.0, 1.0, 0.0, 0.0, 0.0]
    )
    assert rows["action.right_arm.pose.arm_local"][1] is None
    assert not any("shared_bimanual" in column for column in sidecar.column_names)

    parquet_metadata = sidecar.schema.metadata
    assert parquet_metadata[b"abc_geometry.schema_version"].decode() == SIDECAR_SCHEMA_VERSION
    assert (
        parquet_metadata[b"abc_geometry.quaternion_convention"].decode()
        == "Hamilton scalar-first (wxyz)"
    )
    assert parquet_metadata[b"abc_geometry.source_modified"].decode() == "false"

    persisted_manifest = json.loads(manifest_path.read_text())
    assert persisted_manifest == manifest.to_dict()
    assert persisted_manifest["source"]["modified"] is False
    assert persisted_manifest["source"]["dataset_id"] == "lerobot/abc_130k_v3_val"
    assert persisted_manifest["geometry"]["pose_layout"] == [
        "x_m",
        "y_m",
        "z_m",
        "qw",
        "qx",
        "qy",
        "qz",
    ]
    assert "shared_bimanual" in persisted_manifest["geometry"]["excluded_frames"]
    assert persisted_manifest["geometry"]["invalid_input_counts"]["action.right_arm"] == 1
    assert all(
        count == 0 for count in persisted_manifest["geometry"]["derivation_failure_counts"].values()
    )
    assert hashlib.sha256(output.read_bytes()).hexdigest() == manifest.output_sha256


def test_directory_input_preserves_per_file_row_provenance(tmp_path):
    source_dir = tmp_path / "data"
    source_dir.mkdir()
    first = source_dir / "a.parquet"
    second = source_dir / "b.parquet"
    _write_source(first)
    _write_source(second)
    output = tmp_path / "geometry.parquet"

    manifest = generate_pose_sidecar(
        source_dir,
        output,
        kinematics=FakeKinematics(),
        hash_sources=False,
    )

    rows = pq.read_table(output).to_pydict()
    assert manifest.rows == 4
    assert rows["provenance.source_uri"] == [
        first.resolve().as_uri(),
        first.resolve().as_uri(),
        second.resolve().as_uri(),
        second.resolve().as_uri(),
    ]
    assert rows["provenance.source_row"] == [0, 1, 0, 1]
    assert rows["provenance.source_sha256"] == [None, None, None, None]


def test_lerobot_root_reads_only_data_parquets(tmp_path):
    dataset_root = tmp_path / "dataset"
    data_dir = dataset_root / "data" / "chunk-000"
    metadata_dir = dataset_root / "meta" / "episodes" / "chunk-000"
    data_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    source = data_dir / "file-000.parquet"
    _write_source(source)
    # Real LeRobot roots also contain metadata Parquets with a different schema.
    pq.write_table(
        pa.table({"episode_index": [4], "task_index": [9]}), metadata_dir / "file-000.parquet"
    )

    output = tmp_path / "geometry.parquet"
    manifest = generate_pose_sidecar(
        dataset_root,
        output,
        kinematics=FakeKinematics(),
        hash_sources=False,
    )

    assert manifest.rows == 2
    assert len(manifest.source_files) == 1
    assert manifest.source_files[0].uri == source.resolve().as_uri()


def test_info_json_verifies_feature_names_and_is_recorded(tmp_path):
    source = tmp_path / "source.parquet"
    output = tmp_path / "sidecar.parquet"
    info = tmp_path / "info.json"
    _write_source(source)
    _write_info(info)

    manifest = generate_pose_sidecar(
        source,
        output,
        kinematics=FakeKinematics(),
        source_info_path=info,
        source_info_uri="hf://datasets/example@revision/meta/info.json",
    )

    assert manifest.feature_contract["status"] == "verified_info_json"
    assert manifest.feature_contract["uri"] == "hf://datasets/example@revision/meta/info.json"
    assert len(manifest.feature_contract["sha256"]) == 64


def test_mismatched_info_json_fails_before_output(tmp_path):
    source = tmp_path / "source.parquet"
    output = tmp_path / "sidecar.parquet"
    info = tmp_path / "info.json"
    _write_source(source)
    _write_info(info, corrupt_action_names=True)

    with pytest.raises(SidecarSchemaError, match="action names do not match"):
        generate_pose_sidecar(
            source,
            output,
            kinematics=FakeKinematics(),
            source_info_path=info,
        )

    assert not output.exists()


def test_refuses_to_overwrite_source_or_existing_output(tmp_path):
    source = tmp_path / "source.parquet"
    _write_source(source)

    with pytest.raises(SidecarError, match="source input"):
        generate_pose_sidecar(source, source, kinematics=FakeKinematics(), overwrite=True)

    output = tmp_path / "sidecar.parquet"
    output.write_bytes(b"keep me")
    with pytest.raises(FileExistsError, match="output already exists"):
        generate_pose_sidecar(source, output, kinematics=FakeKinematics())
    assert output.read_bytes() == b"keep me"


def test_refuses_outputs_inside_directory_input(tmp_path):
    source_dir = tmp_path / "dataset"
    source_dir.mkdir()
    _write_source(source_dir / "source.parquet")

    with pytest.raises(SidecarError, match="sidecar inside an input directory"):
        generate_pose_sidecar(
            source_dir,
            source_dir / "derived.parquet",
            kinematics=FakeKinematics(),
        )

    with pytest.raises(SidecarError, match="manifest inside an input directory"):
        generate_pose_sidecar(
            source_dir,
            tmp_path / "derived.parquet",
            manifest_path=source_dir / "manifest.json",
            kinematics=FakeKinematics(),
        )


def test_refuses_to_overwrite_info_json_with_manifest_even_with_overwrite(tmp_path):
    source = tmp_path / "source.parquet"
    info = tmp_path / "info.json"
    output = tmp_path / "derived.parquet"
    _write_source(source)
    _write_info(info)
    original_info = info.read_bytes()

    with pytest.raises(SidecarError, match="source input"):
        generate_pose_sidecar(
            source,
            output,
            manifest_path=info,
            source_info_path=info,
            kinematics=FakeKinematics(),
            overwrite=True,
        )

    assert info.read_bytes() == original_info
    assert not output.exists()


def test_missing_required_columns_fail_before_any_output_is_written(tmp_path):
    source = tmp_path / "bad.parquet"
    output = tmp_path / "sidecar.parquet"
    pq.write_table(pa.table({"episode_index": [1]}), source)

    with pytest.raises(SidecarSchemaError, match="missing required columns"):
        generate_pose_sidecar(source, output, kinematics=FakeKinematics())

    assert not output.exists()


def test_malformed_vector_is_key_aligned_but_marked_invalid(tmp_path):
    source = tmp_path / "short-vector.parquet"
    output = tmp_path / "sidecar.parquet"
    table = pa.table(
        {
            "episode_index": [0],
            "frame_index": [0],
            "index": [0],
            "timestamp": [0.0],
            "observation.state": pa.array([[0.0] * 13], type=pa.list_(pa.float64())),
            "action": pa.array([_vector(0, 10)], type=pa.list_(pa.float64())),
        }
    )
    pq.write_table(table, source)

    generate_pose_sidecar(source, output, kinematics=FakeKinematics())

    rows = pq.read_table(output).to_pydict()
    assert rows["geometry.valid_mask"] == [0b1100]
    assert rows["geometry.invalid_input_mask"] == [0b0011]
    assert rows["geometry.derivation_failure_mask"] == [0]
    assert rows["observation.left_arm.pose.arm_local"] == [None]
    assert rows["observation.right_arm.pose.arm_local"] == [None]


def test_fk_failure_is_distinct_from_invalid_source_input(tmp_path):
    source = tmp_path / "source.parquet"
    output = tmp_path / "sidecar.parquet"
    _write_source(source)

    class BrokenRightKinematics(FakeKinematics):
        def grasp_pose(self, joints, *, arm, frame):
            if arm == "right":
                raise RuntimeError("synthetic FK failure")
            return super().grasp_pose(joints, arm=arm, frame=frame)

    manifest = generate_pose_sidecar(source, output, kinematics=BrokenRightKinematics())
    rows = pq.read_table(output).to_pydict()

    assert rows["geometry.valid_mask"] == [0b0101, 0b0101]
    assert rows["geometry.invalid_input_mask"] == [0, 0]
    assert rows["geometry.derivation_failure_mask"] == [0b1010, 0b1010]
    assert manifest.derivation_failure_counts == {
        "observation.left_arm": 0,
        "observation.right_arm": 2,
        "action.left_arm": 0,
        "action.right_arm": 2,
    }


def test_concurrent_source_change_aborts_before_publication(tmp_path):
    source = tmp_path / "source.parquet"
    output = tmp_path / "sidecar.parquet"
    manifest = tmp_path / "sidecar.manifest.json"
    _write_source(source)

    class TouchingKinematics(FakeKinematics):
        touched = False

        def grasp_pose(self, joints, *, arm, frame):
            if not self.touched:
                before = source.stat()
                os.utime(
                    source,
                    ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
                )
                self.touched = True
            return super().grasp_pose(joints, arm=arm, frame=frame)

    with pytest.raises(SidecarError, match="source changed"):
        generate_pose_sidecar(
            source,
            output,
            manifest_path=manifest,
            kinematics=TouchingKinematics(),
        )

    assert not output.exists()
    assert not manifest.exists()


def test_manifest_is_staged_before_parquet_is_published(monkeypatch, tmp_path):
    source = tmp_path / "source.parquet"
    output = tmp_path / "sidecar.parquet"
    manifest = tmp_path / "sidecar.manifest.json"
    _write_source(source)

    def fail_to_stage(*args, **kwargs):
        raise OSError("synthetic manifest staging failure")

    monkeypatch.setattr(sidecar_module, "_stage_json", fail_to_stage)
    with pytest.raises(OSError, match="staging failure"):
        generate_pose_sidecar(
            source,
            output,
            manifest_path=manifest,
            kinematics=FakeKinematics(),
        )

    assert not output.exists()
    assert not manifest.exists()


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        ("episode_index", pa.array([1.5], type=pa.float64()), "integer Arrow type"),
        ("timestamp", pa.array(["nope"]), "numeric Arrow type"),
        (
            "observation.state",
            pa.array([["bad"] * 14], type=pa.list_(pa.string())),
            "list values must be numeric",
        ),
    ],
)
def test_rejects_incompatible_arrow_types(tmp_path, column, values, message):
    source = tmp_path / "bad-types.parquet"
    output = tmp_path / "sidecar.parquet"
    data = {
        "episode_index": pa.array([1], type=pa.int64()),
        "frame_index": pa.array([0], type=pa.int64()),
        "index": pa.array([0], type=pa.int64()),
        "timestamp": pa.array([0.0], type=pa.float64()),
        "observation.state": pa.array([_vector(0, 10)], type=pa.list_(pa.float64())),
        "action": pa.array([_vector(0, 10)], type=pa.list_(pa.float64())),
    }
    data[column] = values
    pq.write_table(pa.table(data), source)

    with pytest.raises(SidecarSchemaError, match=message):
        generate_pose_sidecar(source, output, kinematics=FakeKinematics())

    assert not output.exists()


def test_null_key_aborts_atomically(tmp_path):
    source = tmp_path / "null-key.parquet"
    output = tmp_path / "sidecar.parquet"
    table = pa.table(
        {
            "episode_index": pa.array([None], type=pa.int64()),
            "frame_index": [0],
            "index": [0],
            "timestamp": [0.0],
            "observation.state": pa.array([_vector(0, 10)]),
            "action": pa.array([_vector(0, 10)]),
        }
    )
    pq.write_table(table, source)

    with pytest.raises(SidecarSchemaError, match="episode_index is null"):
        generate_pose_sidecar(source, output, kinematics=FakeKinematics())

    assert not output.exists()
    assert list(tmp_path.glob(".*.tmp.parquet")) == []
