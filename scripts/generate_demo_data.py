"""Generate the demo's independent recorded/FK canary series from raw MCAP."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from abc_geometry.kinematics import YAMKinematics, model_sha256

TOPICS = {
    "/left-arm-state": "left",
    "/right-arm-state": "right",
}


def _validate_provenance(*, source_uri: str, task: str, episode: str) -> None:
    if not source_uri.strip() or not task.strip() or not episode.strip():
        raise ValueError("source URI, task, and episode must be non-empty")
    path_parts = set(Path(unquote(urlparse(source_uri).path)).parts)
    if task not in path_parts or not ({episode, f"episode_{episode}"} & path_parts):
        raise ValueError("task/episode do not match path segments in the canonical source URI")


def build_demo_payload(
    path: Path,
    *,
    source_uri: str,
    task: str,
    episode: str,
    stride: int,
) -> dict:
    if stride <= 0:
        raise ValueError("stride must be positive")
    _validate_provenance(source_uri=source_uri, task=task, episode=episode)

    by_arm: dict[str, list[dict]] = {"left": [], "right": []}
    kinematics = YAMKinematics()
    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, channel, message, decoded in reader.iter_decoded_messages(topics=list(TOPICS)):
            arm = TOPICS[channel.topic]
            joints = np.asarray(decoded.position, dtype=np.float64)
            recorded = np.asarray(decoded.pose, dtype=np.float64)
            if joints.shape != (6,) or recorded.shape != (16,):
                continue
            matrix = recorded.reshape(4, 4)
            derived = kinematics.grasp_pose(
                joints,
                arm=arm,
                frame="shared_bimanual",
            )
            by_arm[arm].append(
                {
                    "log_time_ns": int(message.log_time),
                    "joints_rad": joints.tolist(),
                    "recorded_xyz_m": matrix[:3, 3].tolist(),
                    "derived_xyz_m": derived.position.tolist(),
                    "translation_residual_m": float(
                        np.linalg.norm(matrix[:3, 3] - derived.position)
                    ),
                }
            )

    if not all(by_arm.values()):
        raise RuntimeError("both left and right state topics need intact canary records")

    sampled = {arm: records[::stride] for arm, records in by_arm.items()}
    return {
        "schema_version": "abc-geometry-demo-canary/v1",
        "source": {
            "uri": source_uri,
            "task": task,
            "episode": episode,
        },
        "derivation": {
            "model_sha256": model_sha256(),
            "frame": "shared_bimanual",
            "stride_records": stride,
        },
        "series": sampled,
        "validation": {
            arm: {
                "source_records": len(by_arm[arm]),
                "sampled_records": len(sampled[arm]),
                "max_translation_residual_m": max(
                    row["translation_residual_m"] for row in by_arm[arm]
                ),
            }
            for arm in ("left", "right")
        },
    }


def _write_json_atomic(path: Path, payload: dict, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists (pass --overwrite): {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mcap", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.mcap.resolve() == args.output.resolve():
        raise ValueError("refusing to overwrite the input MCAP with demo JSON")
    payload = build_demo_payload(
        args.mcap,
        source_uri=args.source_uri,
        task=args.task,
        episode=args.episode,
        stride=args.stride,
    )
    _write_json_atomic(args.output, payload, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
