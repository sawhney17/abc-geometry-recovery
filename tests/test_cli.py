from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from abc_geometry import cli
from abc_geometry.mcap_validation import MCAPAuditReport, _new_report


def _empty_report(source: str, *, passed: bool) -> MCAPAuditReport:
    report = _new_report(source, 1e-5, 1e-3, 100)
    if passed:
        for topic in report.topics.values():
            topic.messages = 1
            topic.intact_pose = 1
    return report


def test_audit_cli_writes_combined_machine_readable_report(monkeypatch, tmp_path):
    source = tmp_path / "episode.mcap"
    output = tmp_path / "audit.json"
    source.write_bytes(b"unused by fake")
    monkeypatch.setattr(
        cli,
        "audit_mcap",
        lambda *args, **kwargs: _empty_report(str(source), passed=True),
    )

    status = cli.main(["audit-mcap", str(source), "--report", str(output)])

    assert status == 0
    payload = json.loads(output.read_text())
    assert payload["passed"] is True
    assert payload["totals"]["files"] == 1
    assert payload["totals"]["target_messages"] == 4


def test_audit_cli_returns_one_for_a_data_failure(monkeypatch, tmp_path, capsys):
    source = tmp_path / "episode.mcap"
    source.write_bytes(b"unused by fake")
    monkeypatch.setattr(
        cli,
        "audit_mcap",
        lambda *args, **kwargs: _empty_report(str(source), passed=False),
    )

    status = cli.main(["audit-mcap", str(source)])

    assert status == 1
    assert json.loads(capsys.readouterr().out)["passed"] is False


def test_cli_refuses_to_replace_existing_report(monkeypatch, tmp_path):
    source = tmp_path / "episode.mcap"
    output = tmp_path / "audit.json"
    source.write_bytes(b"unused by fake")
    output.write_text("keep")
    monkeypatch.setattr(
        cli,
        "audit_mcap",
        lambda *args, **kwargs: _empty_report(str(source), passed=True),
    )

    status = cli.main(["audit-mcap", str(source), "--report", str(output)])

    assert status == 2
    assert output.read_text() == "keep"


def test_audit_cli_never_overwrites_an_input_mcap(monkeypatch, tmp_path):
    source = tmp_path / "episode.mcap"
    original = b"irreplaceable source bytes"
    source.write_bytes(original)

    called = False

    def fake_audit(*args, **kwargs):
        nonlocal called
        called = True
        return _empty_report(str(source), passed=True)

    monkeypatch.setattr(cli, "audit_mcap", fake_audit)
    status = cli.main(["audit-mcap", str(source), "--report", str(source), "--overwrite"])

    assert status == 2
    assert called is False
    assert source.read_bytes() == original


def test_recover_mcap_cli_aligns_and_forwards_canonical_source_metadata(
    monkeypatch,
    tmp_path,
    capsys,
):
    first = tmp_path / "first.mcap"
    second = tmp_path / "second.mcap"
    output = tmp_path / "recovered.parquet"
    manifest_path = tmp_path / "public.manifest.json"
    captured = {}

    class Manifest:
        rows = 2

        def to_dict(self):
            return {"schema_version": "test", "output": {"rows": self.rows}}

    def fake_generate(sources, output_path, **kwargs):
        captured["sources"] = sources
        captured["output"] = output_path
        captured["kwargs"] = kwargs
        return Manifest()

    monkeypatch.setattr(cli, "generate_raw_pose_sidecar", fake_generate)
    status = cli.main(
        [
            "recover-mcap",
            str(first),
            str(second),
            "--output",
            str(output),
            "--manifest",
            str(manifest_path),
            "--source-uri",
            "hf://dataset@commit/first.mcap",
            "--source-uri",
            "hf://dataset@commit/second.mcap",
            "--task",
            "task-a",
            "--task",
            "task-b",
            "--episode",
            "episode-a",
            "--episode",
            "episode-b",
            "--source-sha256",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--source-sha256",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "--source-size-bytes",
            "101",
            "--source-size-bytes",
            "202",
            "--dataset-id",
            "example/abc",
            "--dataset-revision",
            "commit",
            "--output-uri",
            "hf://release/recovered.parquet",
            "--manifest-uri",
            "hf://release/recovered.manifest.json",
            "--row-group-size",
            "17",
            "--overwrite",
        ]
    )

    assert status == 0
    assert [source.path for source in captured["sources"]] == [first, second]
    assert [source.canonical_uri for source in captured["sources"]] == [
        "hf://dataset@commit/first.mcap",
        "hf://dataset@commit/second.mcap",
    ]
    assert [source.task for source in captured["sources"]] == ["task-a", "task-b"]
    assert [source.episode for source in captured["sources"]] == ["episode-a", "episode-b"]
    assert [source.expected_sha256 for source in captured["sources"]] == [
        "a" * 64,
        "b" * 64,
    ]
    assert [source.expected_size_bytes for source in captured["sources"]] == [101, 202]
    assert captured["output"] == output
    assert captured["kwargs"] == {
        "manifest_path": manifest_path,
        "dataset_id": "example/abc",
        "dataset_revision": "commit",
        "output_uri": "hf://release/recovered.parquet",
        "manifest_uri": "hf://release/recovered.manifest.json",
        "row_group_size": 17,
        "overwrite": True,
    }
    terminal = capsys.readouterr()
    assert json.loads(terminal.out)["output"]["rows"] == 2
    assert "recovered 2 raw arm messages" in terminal.err


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--source-uri", "only-one-value"),
        ("--task", "only-one-value"),
        ("--episode", "only-one-value"),
        ("--source-sha256", "a" * 64),
        ("--source-size-bytes", "1"),
    ],
)
def test_recover_mcap_cli_rejects_partial_per_source_metadata(
    option,
    value,
    monkeypatch,
    tmp_path,
    capsys,
):
    called = False

    def fake_generate(*args, **kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(rows=0, to_dict=lambda: {})

    monkeypatch.setattr(cli, "generate_raw_pose_sidecar", fake_generate)
    status = cli.main(
        [
            "recover-mcap",
            str(tmp_path / "first.mcap"),
            str(tmp_path / "second.mcap"),
            "--output",
            str(tmp_path / "output.parquet"),
            option,
            value,
        ]
    )

    assert status == 2
    assert called is False
    assert f"{option} must be repeated exactly once per source MCAP" in capsys.readouterr().err


def test_recover_mcap_cli_allows_all_per_source_metadata_to_be_inferred(
    monkeypatch,
    tmp_path,
):
    captured = {}

    class Manifest:
        rows = 0

        def to_dict(self):
            return {}

    def fake_generate(sources, output_path, **kwargs):
        captured["sources"] = sources
        captured["kwargs"] = kwargs
        return Manifest()

    monkeypatch.setattr(cli, "generate_raw_pose_sidecar", fake_generate)
    output = tmp_path / "output.parquet"
    status = cli.main(["recover-mcap", str(tmp_path / "source.mcap"), "--output", str(output)])

    assert status == 0
    assert captured["sources"][0].canonical_uri is None
    assert captured["sources"][0].task is None
    assert captured["sources"][0].episode is None
    assert captured["sources"][0].expected_sha256 is None
    assert captured["sources"][0].expected_size_bytes is None
    assert captured["kwargs"]["manifest_path"] == tmp_path / "output.manifest.json"
