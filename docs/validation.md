# Validation record

## Complete public raw scan

The primary evidence is the complete public validation tree in
[`Voxel51/ABC-130k`](https://huggingface.co/datasets/Voxel51/ABC-130k), pinned to revision
`9659e8ce4b39580f48369cc31bc2e47a217c40e7`. The scope is exactly every
`data/val/**/episode.fo.mcap` file present at that revision: **40 files**, one episode per file,
totalling **3,509,376,768 bytes**. This is the whole ungated public slice, not a claim about the
official gated ABC-130K corpus.

Every source path, byte size, Git blob ID, and Git-LFS object digest is recorded in
[`../evidence/public-40-sources.tsv`](../evidence/public-40-sources.tsv); canonical source URIs are
formed from those paths and the pinned dataset revision above. The compact result and
release-artifact provenance are in
[`../evidence/public-40-summary.json`](../evidence/public-40-summary.json).

The scan counted only the four exact raw protobuf topics below:

- `/left-arm-state`
- `/right-arm-state`
- `/left-arm-action`
- `/right-arm-action`

Voxel51's `.fo.mcap` files also contain `.plot` messages derived from those streams. They were
excluded so that no arm sample was counted twice.

## Missingness and recoverability

Across **641,593** target arm records, pose presence is binary: a record has either an empty
protobuf pose field or a complete 16-value matrix. There are no partial-length, malformed, or
non-finite poses in this slice.

| Result | Count |
|---|---:|
| Empty/missing pose records | 314,081 (48.95%) |
| Intact 16-value pose records | 327,512 (51.05%) |
| Malformed or non-finite pose records | 0 |
| Records with exactly six finite joints | 641,593 |
| Records with invalid joints | 0 |
| Episodes with every target pose missing | 8 |
| Episodes with every target pose intact | 32 |
| Episodes with mixed pose presence | 0 |

The eight affected episodes are fully affected: every record on all four target arm topics has an
empty pose field. Every one of the **314,081 missing-pose records** retains exactly six finite joint
positions and successfully produces a finite `grasp_site` pose through the versioned YAM model.
They are therefore all recoverable in the arm-local frame. Their original shared-bimanual frame
cannot be reconstructed safely without knowing which physical rig calibration produced them.

### Independent protobuf cross-check

A second scanner independently reopened all 40 MCAPs without using
`abc_geometry.mcap_validation`'s pose or joint classification helpers. It literally counted the
length and finiteness of the decoded protobuf `pose` and `position` fields, rehashed every input,
and compared its totals with the primary audit. It found only pose lengths 0 and 16, only joint
length 6, and reproduced all **314,081 missing**, **327,512 intact**, and **641,593 total** counts
exactly. The cross-check passed every source-hash and count comparison; its release filename and
digest are recorded in the public-40 summary.

## FK validation and frame cohorts

The **32 intact episodes** provide 327,512 independent recorded transforms against which to test
the official YAM `grasp_site` forward kinematics. Left-arm poses match arm-local FK directly.
Right-arm recorded positions reveal two distinct base-translation cohorts:

| Observed right-base translation | Episodes | Intact right-arm records |
|---|---:|---:|
| `[0, -0.61, 0]` m | 30 | 147,638 |
| `[0, -0.80, 0]` m | 2 | 16,118 |

After applying each intact episode's observed cohort offset, the maximum translation residual is
`6.667118051786499e-16 m` and the maximum rotation residual is
`4.517745487845102e-06°`. Position calculations use double precision; the rotation residual is a
geodesic angle whose few-millionths-of-a-degree floor is expected numerical behavior near
identity.

This cohort split matters operationally. Hard-coding the more common `-0.61 m` transform makes all
16,118 right-arm records from the `-0.80 m` cohort appear to have a **0.19 m (19 cm) FK error**,
even though their kinematics are correct. The validator therefore reports observed per-topic base
translation statistics separately from fixed shared-frame FK residuals. The eight missing-pose
episodes receive no shared-frame cohort label.

## Published raw recovery

Release `v0.1.0` contains a combined sidecar for the eight fully affected public episodes.

| Check | Result |
|---|---:|
| Source files | 8 |
| Output rows | 314,081 |
| Output bytes | 17,742,217 |
| Output SHA-256 | `511bc785df6df8a080e97530f991669f3527444a2bcfddb4816846f0bb97bcb6` |
| Source pose status | 314,081 missing |
| Joint status | 314,081 intact |
| FK status | 314,081 derived |
| Recovery status | 314,081 recovered |
| Non-finite output poses | 0 |
| Maximum quaternion norm error | 2.22e-16 |

Every source's expected Hub LFS SHA-256 and byte size were checked before decoding, then its local
identity and hash were checked again after all FK work. All rows use status mask `14` (intact
joints, valid FK, and recovered source pose), and every pose is a finite fixed-size seven-value
arm-local vector. A second clean generation produced byte-identical Parquet. The exact source,
model, schema, status counts, public artifact URI, and output digest are recorded in the
[`raw recovery manifest`](../evidence/abc-130k-public-missing-pose-recovery.manifest.json).

## Secondary: published LeRobot augmentation

Release `v0.1.0` compiles `data/chunk-000/file-000.parquet` from
`lerobot/abc_130k_v3_train` revision
`68651e4929d9fb00f798937b2d62617cab5c771d`.

| Check | Result |
|---|---:|
| Input rows | 384,567 |
| Episodes | 127 (indices 0–126) |
| Input SHA-256 | `919e4b62ab2c154e242436a818cc3589fa3880f555fef5448a22e06f8238d227` |
| Info contract SHA-256 | `6c28ba76ecd04adcd3f198208c3a588ef18ab46f4f30f31897d7d90d4243914b` |
| Output bytes | 81,446,405 |
| Output SHA-256 | `268b26276a7fb345017c96b25db4eb586a70c05c2c1fc9c07741b32629ff3da9` |
| Arm-local pose columns | 4 |
| Derived pose vectors | 1,538,268 |
| Rows with all four validity bits | 384,567 |
| Invalid-input bits | 0 |
| FK-failure bits | 0 |
| Maximum quaternion norm error | 2.22e-16 |
| End-to-end wall time | 38.2 s on one local CPU process |

The source file is opened read-only, hashed before and after derivation, and never changed. The
result contains observed/commanded × left/right arm-local poses. It deliberately excludes the
empirical shared-bimanual transform because the official converted shard does not identify its rig
cohort. See the [`sidecar manifest`](../evidence/abc-geometry-file-000.manifest.json) for exact
provenance and mask assignments.

Two independent full generations produced the same Parquet SHA-256. The manifest timestamp is
expected to differ between runs; the data artifact is deterministic.

## What this does and does not prove

It proves that:

- 314,081 arm records in the complete 40-episode public slice have empty pose fields while
  retaining valid joints;
- all 314,081 missing poses are deterministically recoverable in the arm-local frame;
- the official public chain, joint order, units, and target site reproduce geometry across all 32
  intact episodes;
- the intact episodes contain at least two bimanual-frame translation conventions, not one;
- both observed and commanded arm-local poses are deterministically derivable from the published
  LeRobot joint vectors;
- the published LeRobot slice can be augmented without mutating or ambiguously interpreting it.

It does not prove that:

- 60,982 episodes in the full corpus are missing pose—that separate count comes from the author of
  [issue #13](https://github.com/amazon-far/abc/issues/13), and the official raw corpus is gated;
- the original shared-bimanual pose of any of the eight affected public episodes is known;
- any of the 127 published augmentation episodes were broken;
- camera extrinsics can be recovered from arm joints alone;
- adding end-effector pose improves an existing joint-space training recipe.

Applying the same content-addressed scan to the official gated corpus is the next prevalence gate.
Any full-corpus repair release should preserve the same evidence standard: observed missingness,
valid-joint coverage among missing records, residual distributions by rig cohort, explicit rejected
coverage, and an exact dataset revision.
