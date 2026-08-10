from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy

import pytest

from scripts import audit_public_preview as preview


def _pinned_fixture(monkeypatch, tmp_path):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    tree = []
    samples = []
    for index in range(preview.EXPECTED_FILE_COUNT):
        task = f"task_{index:02d}"
        episode = f"episode_{index:02d}"
        relative = f"data/val/{task}/{episode}/episode.fo.mcap"
        data = f"source-{index}".encode()
        source = source_root / relative
        source.parent.mkdir(parents=True)
        source.write_bytes(data)
        tree.append(
            {
                "type": "file",
                "path": relative,
                "size": len(data),
                "oid": hashlib.sha1(f"pointer-{index}".encode(), usedforsecurity=False).hexdigest(),
                "lfs": {
                    "oid": hashlib.sha256(data).hexdigest(),
                    "size": len(data),
                },
            }
        )
        samples.append(
            {
                "filepath": relative,
                "split": "val",
                "task": task,
                "episode_id": episode,
                "n_messages": 8,
            }
        )
    samples_payload = {"samples": samples}
    samples_bytes = json.dumps(samples_payload, sort_keys=True).encode()
    sample_oid = preview.git_blob_oid(samples_bytes)
    tree.append(
        {
            "type": "file",
            "path": "samples.json",
            "size": len(samples_bytes),
            "oid": sample_oid,
        }
    )
    normalised = [preview._normalise_mcap_entry(entry) for entry in tree[:-1]]
    monkeypatch.setattr(
        preview,
        "PINNED_MCAP_INVENTORY_SHA256",
        preview.mcap_inventory_fingerprint(normalised),
    )
    monkeypatch.setattr(preview, "PINNED_SAMPLES_GIT_BLOB_OID", sample_oid)
    monkeypatch.setattr(preview, "PINNED_SAMPLES_SIZE_BYTES", len(samples_bytes))
    return tree, samples_payload, samples_bytes, source_root


def test_build_jobs_enforces_exact_pinned_inventory(monkeypatch, tmp_path):
    tree, samples, samples_bytes, source_root = _pinned_fixture(monkeypatch, tmp_path)

    jobs = preview.build_jobs(tree, samples, samples_bytes, source_root)

    assert len(jobs) == 40
    assert jobs[0]["hf_uri"].startswith(
        f"hf://datasets/{preview.DATASET_ID}@{preview.DATASET_REVISION}/data/val/"
    )
    assert jobs[0]["tree"]["lfs"]["oid"] == hashlib.sha256(b"source-0").hexdigest()

    altered = deepcopy(tree)
    altered[0]["oid"] = "f" * 40
    with pytest.raises(preview.PublicPreviewAuditError, match="pinned Voxel51 revision"):
        preview.build_jobs(altered, samples, samples_bytes, source_root)


def test_build_jobs_rejects_samples_not_bound_to_tree(monkeypatch, tmp_path):
    tree, samples, samples_bytes, source_root = _pinned_fixture(monkeypatch, tmp_path)

    with pytest.raises(preview.PublicPreviewAuditError, match="byte size"):
        preview.build_jobs(tree, samples, samples_bytes + b"\n", source_root)

    mismatched = deepcopy(samples)
    mismatched["samples"][0]["task"] = "wrong_task"
    mismatched_bytes = json.dumps(mismatched, sort_keys=True).encode()
    monkeypatch.setattr(preview, "PINNED_SAMPLES_SIZE_BYTES", len(mismatched_bytes))
    mismatched_oid = preview.git_blob_oid(mismatched_bytes)
    monkeypatch.setattr(preview, "PINNED_SAMPLES_GIT_BLOB_OID", mismatched_oid)
    tree[-1]["size"] = len(mismatched_bytes)
    tree[-1]["oid"] = mismatched_oid
    with pytest.raises(preview.PublicPreviewAuditError, match="task does not match"):
        preview.build_jobs(tree, mismatched, mismatched_bytes, source_root)


def _offset(count, y, *, minimum=None, maximum=None):
    if count == 0:
        return {
            "count": 0,
            "mean_xyz": None,
            "std_xyz": None,
            "min_xyz": None,
            "max_xyz": None,
        }
    return {
        "count": count,
        "mean_xyz": [0.0, y, 0.0],
        "std_xyz": [0.0, 0.0, 0.0],
        "min_xyz": [0.0, y if minimum is None else minimum, 0.0],
        "max_xyz": [0.0, y if maximum is None else maximum, 0.0],
    }


def test_combines_final_validator_offset_statistics_without_rescan():
    combined = preview.combine_offset_summaries(
        [
            _offset(1, -0.61),
            _offset(3, -0.63),
        ]
    )

    assert combined["compared_pose_records"] == 4
    assert combined["translation_m"]["mean"] == pytest.approx([0.0, -0.625, 0.0])
    assert combined["translation_m"]["std"] == pytest.approx([0.0, math.sqrt(0.000075), 0.0])
    assert combined["translation_m"]["min"] == pytest.approx([0.0, -0.63, 0.0])
    assert combined["translation_m"]["max"] == pytest.approx([0.0, -0.61, 0.0])


def _topic_payload(topic, *, missing, right_offset):
    intact = 0 if missing else 1
    compared = intact
    arm = preview.ARM_TOPICS[topic][0]
    offset = _offset(compared, right_offset if arm == "right" else 0.0)
    return {
        "arm": arm,
        "stream": preview.ARM_TOPICS[topic][1],
        "messages": 1,
        "poses": {
            "missing": int(missing),
            "malformed": 0,
            "nonfinite": 0,
            "intact": intact,
            "recoverable_invalid": int(missing),
            "unrecoverable_invalid": 0,
        },
        "joints": {"missing": 0, "malformed": 0, "nonfinite": 0},
        "fk": {
            "compared": compared,
            "failures": 0,
            "outside_tolerance": 0,
            "translation_error_m": {
                "mean": 0.0 if compared else None,
                "rmse": 0.0 if compared else None,
                "max": 0.0 if compared else None,
            },
            "rotation_error_deg": {
                "mean": 0.0 if compared else None,
                "rmse": 0.0 if compared else None,
                "max": 0.0 if compared else None,
            },
            "observed_base_translation_offset_m": offset,
        },
    }


def _file_payload(index, *, missing=False, right_offset=-0.61):
    path = f"data/val/task_{index:02d}/episode_{index:02d}/episode.fo.mcap"
    hf_uri = f"hf://datasets/{preview.DATASET_ID}@{preview.DATASET_REVISION}/{path}"
    topics = {
        topic: _topic_payload(topic, missing=missing, right_offset=right_offset)
        for topic in preview.ARM_TOPICS
    }
    invalid = 4 if missing else 0
    compared = 0 if missing else 4
    audit = {
        "source": hf_uri,
        "passed": not missing,
        "totals": {
            "target_messages": 4,
            "invalid_poses": invalid,
            "intact_poses": 4 - invalid,
            "recoverable_invalid_poses": invalid,
            "unrecoverable_invalid_poses": 0,
            "fk_compared": compared,
            "fk_outside_tolerance": 0,
            "decode_errors": 0,
        },
        "missing_topics": [],
        "topics": topics,
        "issue_count": invalid,
    }
    return {
        "path": path,
        "hf_uri": hf_uri,
        "task": f"task_{index:02d}",
        "episode_id": f"episode_{index:02d}",
        "sample_metadata": {"declared_messages": 4},
        "object": {
            "size_bytes_before": 1,
            "size_matches_manifest": True,
            "sha256_matches_lfs": True,
            "read_only_unchanged": True,
        },
        "mcap_summary": {
            "all_messages": 8,
            "generated_plot_messages": 4,
            "target_topic_counts": {topic: 1 for topic in preview.ARM_TOPICS},
        },
        "observed_right_arm_base_offset": preview.observed_right_arm_base_offset(audit),
        "audit": audit,
    }


def test_aggregation_finds_cohorts_and_plot_invariants():
    files = [
        _file_payload(
            index,
            missing=index >= 32,
            right_offset=-0.80 if 30 <= index < 32 else -0.61,
        )
        for index in range(40)
    ]

    result = preview.aggregate(files)
    totals = result["totals"]
    clusters = result["right_arm_base_offset_analysis"]

    assert totals["target_messages"] == 160
    assert totals["poses"]["missing"] == 32
    assert totals["poses"]["intact"] == 128
    assert clusters["files_with_observable_right_arm_offset"] == 32
    assert clusters["files_without_observable_right_arm_offset"] == 8
    assert [
        (cluster["translation_m_rounded"], cluster["files"], cluster["pose_records"])
        for cluster in clusters["clusters"]
    ] == [
        ([0.0, -0.8, 0.0], 2, 4),
        ([0.0, -0.61, 0.0], 30, 60),
    ]
    assert all(check["passed"] for check in preview.validate_invariants(files, totals))

    files[0]["mcap_summary"]["all_messages"] += 1
    checks = {
        check["name"]: check["passed"] for check in preview.validate_invariants(files, totals)
    }
    assert not checks["samples_declared_plus_generated_plot_counts_match_mcap_summaries"]


def test_source_hash_guard_detects_mutation(tmp_path):
    source = tmp_path / "episode.fo.mcap"
    source.write_bytes(b"original")
    tree_entry = {
        "size": len(b"original"),
        "lfs": {"oid": hashlib.sha256(b"original").hexdigest()},
    }
    stat, digest = preview._verify_source_before(source, tree_entry)

    source.write_bytes(b"modified")

    with pytest.raises(preview.PublicPreviewAuditError, match="pinned LFS object"):
        preview._verify_source_after(source, tree_entry, stat, digest)


def test_output_guards_and_atomic_failure_preserve_inputs(tmp_path):
    tree = tmp_path / "tree.json"
    samples = tmp_path / "samples.json"
    source_root = tmp_path / "sources"
    tree.write_text("[]")
    samples.write_text("{}")
    source_root.mkdir()

    with pytest.raises(preview.PublicPreviewAuditError, match="inside --source-root"):
        preview.guard_output_path(
            source_root / "audit.json",
            tree_path=tree,
            samples_path=samples,
            source_root=source_root,
            overwrite=False,
        )
    with pytest.raises(preview.PublicPreviewAuditError, match="input manifest"):
        preview.guard_output_path(
            tree,
            tree_path=tree,
            samples_path=samples,
            source_root=source_root,
            overwrite=True,
        )

    output = tmp_path / "audit.json"
    output.write_text("keep")
    with pytest.raises(FileExistsError, match="already exists"):
        preview.write_json_atomic(output, {"replacement": True}, overwrite=False)
    with pytest.raises(ValueError, match="Out of range float"):
        preview.write_json_atomic(output, {"bad": float("nan")}, overwrite=True)
    assert output.read_text() == "keep"
