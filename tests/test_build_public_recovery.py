from __future__ import annotations

from copy import deepcopy

import pytest

from abc_geometry.raw_recovery import RawRecoveryError
from scripts.build_public_recovery import AUDIT_SCHEMA, select_fully_missing_sources


def _report(*, target=8, invalid=8, recoverable=8, unrecoverable=0):
    topic = {
        "poses": {
            "missing": target // 4,
            "malformed": 0,
            "nonfinite": 0,
            "intact": 0,
        }
    }
    return {
        "schema_version": AUDIT_SCHEMA,
        "files": [
            {
                "path": "data/val/task/episode_id/episode.fo.mcap",
                "hf_uri": "hf://datasets/example/abc@commit/data/val/task/episode_id/episode.fo.mcap",
                "task": "task",
                "episode_id": "episode_id",
                "object": {
                    "lfs_sha256": "a" * 64,
                    "expected_size_bytes": 123,
                },
                "audit": {
                    "totals": {
                        "target_messages": target,
                        "invalid_poses": invalid,
                        "recoverable_invalid_poses": recoverable,
                        "unrecoverable_invalid_poses": unrecoverable,
                    },
                    "topics": {f"topic-{index}": deepcopy(topic) for index in range(4)},
                },
            }
        ],
    }


def test_selects_only_fully_missing_recoverable_episodes(tmp_path):
    sources, rows = select_fully_missing_sources(_report(), tmp_path)

    assert rows == 8
    assert len(sources) == 1
    assert sources[0].path == tmp_path / "data/val/task/episode_id/episode.fo.mcap"
    assert sources[0].canonical_uri.startswith("hf://datasets/example/abc@commit/")
    assert sources[0].task == "task"
    assert sources[0].episode == "episode_id"
    assert sources[0].expected_sha256 == "a" * 64
    assert sources[0].expected_size_bytes == 123


def test_refuses_mixed_or_unrecoverable_release_slices(tmp_path):
    with pytest.raises(RawRecoveryError, match="mixed intact/invalid"):
        select_fully_missing_sources(_report(target=8, invalid=4, recoverable=4), tmp_path)

    with pytest.raises(RawRecoveryError, match="not every invalid pose is recoverable"):
        select_fully_missing_sources(_report(target=8, invalid=8, recoverable=7), tmp_path)


def test_refuses_source_path_traversal(tmp_path):
    report = _report()
    report["files"][0]["path"] = "../outside.mcap"

    with pytest.raises(RawRecoveryError, match="escapes --source-root"):
        select_fully_missing_sources(report, tmp_path)
