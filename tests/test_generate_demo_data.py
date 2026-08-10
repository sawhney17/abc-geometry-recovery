from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_demo_data.py"
_SPEC = spec_from_file_location("generate_demo_data", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
generate_demo_data = module_from_spec(_SPEC)
_SPEC.loader.exec_module(generate_demo_data)

SOURCE_URI = (
    "hf://datasets/Voxel51/ABC-130k@revision/data/val/"
    "fold_and_stack_the_skirts/episode_abc123/episode.fo.mcap"
)


def test_provenance_must_match_canonical_uri_segments():
    generate_demo_data._validate_provenance(
        source_uri=SOURCE_URI,
        task="fold_and_stack_the_skirts",
        episode="abc123",
    )

    with pytest.raises(ValueError, match="do not match"):
        generate_demo_data._validate_provenance(
            source_uri=SOURCE_URI,
            task="a_different_task",
            episode="abc123",
        )


def test_main_never_overwrites_input_mcap(monkeypatch, tmp_path):
    source = tmp_path / "episode.mcap"
    original = b"raw source bytes"
    source.write_bytes(original)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_demo_data.py",
            str(source),
            str(source),
            "--source-uri",
            SOURCE_URI,
            "--task",
            "fold_and_stack_the_skirts",
            "--episode",
            "abc123",
            "--overwrite",
        ],
    )

    with pytest.raises(ValueError, match="input MCAP"):
        generate_demo_data.main()

    assert source.read_bytes() == original


def test_atomic_writer_refuses_existing_output_without_overwrite(tmp_path):
    output = tmp_path / "example.json"
    output.write_text("keep")

    with pytest.raises(FileExistsError, match="already exists"):
        generate_demo_data._write_json_atomic(output, {"replacement": True}, overwrite=False)

    assert output.read_text() == "keep"
