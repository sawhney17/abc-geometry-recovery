# Method and data contract

## Derivation target

ABC raw arm messages expose six revolute joint positions and may expose a 4×4 pose matrix. The
matrix corresponds to the official YAM model's `grasp_site`, not its wrist origin or `tcp_site`.
The compiler derives one narrowly named quantity:

```text
six joint angles (radians) + versioned YAM chain -> grasp_site SE(3) pose
```

Gripper position does not affect this rigid pose and remains only in the source dataset.

## Raw scan and classification

The public audit is content-addressed and revision-bounded. It enumerates every
`data/val/**/episode.fo.mcap` object in `Voxel51/ABC-130k` revision
`9659e8ce4b39580f48369cc31bc2e47a217c40e7`, records its canonical URI, byte size, and SHA-256,
then reads only the four exact arm state/action topics. Derived `.plot` messages are excluded.

Each decoded record is classified before any recovery is attempted:

1. A pose field of length zero is `missing`.
2. A pose must otherwise contain 16 finite values, a valid homogeneous bottom row, an orthonormal
   rotation, and determinant +1. Wrong-length or structurally invalid matrices are `malformed`;
   NaN or infinity is `nonfinite`.
3. A joint field must contain exactly six finite values in source order. Missing, malformed, and
   non-finite joints are counted separately.
4. Every refusal is counted even when the sampled issue log is capped.

An independent cross-check deliberately avoids these classification helpers. It reopens the same
content-addressed MCAPs, counts the literal lengths and finiteness of protobuf `pose` and
`position`, and requires its histograms, totals, and input hashes to match the primary audit.

## FK validation and recovery

Intact records establish the recovery invariant before missing records are filled. For each valid
six-joint vector, the versioned official YAM chain evaluates `grasp_site`. When the kinematics
adapter supports it, arm-local and shared-frame candidates come from one FK evaluation. The audit
then:

1. compares orientation with a clipped geodesic rotation error;
2. computes the observed base translation as
   `recorded_xyz - arm_local_fk_xyz` for each topic;
3. aggregates count, mean, standard deviation, minimum, and maximum XYZ so rig cohorts remain
   visible instead of being collapsed into residual failures;
4. validates intact episodes after applying their observed cohort-specific translation; and
5. derives a missing pose only when all six joints are finite and FK returns a finite pose.

The recovery output for a missing raw record is arm-local FK with explicit validity status. The
source MCAP remains unchanged. Missing records with invalid joints or failed FK would be refused,
not guessed; in the final public-40 scan, all missing records passed the arm-local recovery gate.

## Coordinate frames

The released sidecar emits only `arm_local`: the origin is the base of the relevant arm, and the
same YAM chain is used for the left and right arms.

The raw validator separately measures the observed `shared_bimanual` convention anchored at the
left-arm base. In the full public scan, 30 intact-pose episodes use right-base translation
`[0, -0.61, 0]` metres while 2 use `[0, -0.80, 0]`; rotations remain identity. These transforms
are empirical, not declared by the single-arm YAM MJCF. No shared transform is emitted for the 8
missing-pose episodes or the LeRobot shard because their rig cohorts cannot be inferred safely.

This exclusion is essential, not a missing feature. Joint telemetry plus the single-arm model
determines `T_arm_base_grasp`; it does not determine the extrinsic placement of that arm base in a
shared station. The intact public episodes prove that at least two placements exist. Assigning the
common `-0.61 m` offset to a `-0.80 m` rig creates a systematic 19 cm error while the arm
kinematics remain correct. The missing-pose episodes have no recorded transform from which to
identify their cohort, and the LeRobot conversion does not retain a trustworthy rig identifier.
Recovery artifacts therefore emit only the independently determined arm-local pose and mark
`shared_bimanual` as excluded.

## Pose representation

Sidecar vectors are:

```text
[x_m, y_m, z_m, qw, qx, qy, qz]
```

Quaternions use scalar-first `wxyz` order. The Parquet metadata and JSON manifest record this
order, translation units, exact source feature indices, frame name, model digest, model source
revision, tool version, and dataset revision. The manifest explicitly records
`shared_bimanual` as excluded rather than silently guessing a transform.

## Raw recovery row contract

Raw MCAP recovery emits one row per exact arm state/action message. Its status bits are:

- bit 0: the source pose was structurally intact;
- bit 1: the six-joint input was intact;
- bit 2: FK produced a finite seven-value pose;
- bit 3: a missing/invalid source pose was recovered by valid FK.

The raw `geometry.pose.arm_local` column is a non-null fixed-size seven-float list. When FK is
refused or fails, it contains seven NaNs; `geometry.fk_status`, `geometry.recovery_status`, and the
status mask are authoritative. This avoids variable-length vectors while keeping every rejected
source message key-aligned. The published public-40 repair contains no such rejected vectors: all
314,081 rows have mask `14`, finite poses, and `recovery_status=recovered`.

## Secondary LeRobot row status contract

Bits 0–3 always mean observed-left, observed-right, commanded-left, and commanded-right. Three
mutually exclusive masks make refusal inspectable:

- `geometry.valid_mask`: six finite source joints produced a finite seven-value pose;
- `geometry.invalid_input_mask`: the source vector was null, malformed, or non-finite;
- `geometry.derivation_failure_mask`: the input was valid but FK failed or returned an invalid
  pose.

A pose is null whenever its valid bit is unset. Rows remain key-aligned even when individual poses
are refused.

## Provenance rules

1. Never edit raw MCAP or source LeRobot shards.
2. Pin the dataset revision and inventory every input with canonical URI, byte size, and digest
   before interpreting records.
3. Key every derived row to `episode_index`, `frame_index`, source `index`, timestamp, source URI,
   source row, and source digest.
4. Record the exact kinematic-model source revision and content digest, pose convention, frame,
   joint mapping, tool version, and refusal masks with the output.
5. Mark every current sidecar pose `derived_fk`; this release does not copy or blend raw poses.
6. Validate deterministic output against intact records from the same observed hardware cohort.
7. Hash source files before and after derivation and abort if identity or content changes.
8. Quarantine a cohort when intact residuals exceed declared tolerances.
9. Do not promote SLAM, learned calibration, or inferred rig identity into deterministic fields.

Single JSON reports are written to a temporary file in the destination directory, flushed, and
atomically renamed. Paired recovery/augmentation publications stage both Parquet and manifest,
then atomically rename the Parquet first and the manifest last. The manifest is the commit marker:
consumers verify that the Parquet digest and provenance recorded in it match the visible artifact.
No filesystem provides a cross-file atomic rename, so a crash between the two renames is detected
as a missing or stale manifest rather than accepted as a complete publication. Rerunning with
`--overwrite` repairs the pair.

## Secondary LeRobot augmentation contract

The current compiler pass is intentionally specific to the published 14-D ABC conversion:

| Output | Source column | Joint indices | Units |
|---|---|---:|---|
| Observed left pose | `observation.state` | 0–5 | radians |
| Observed right pose | `observation.state` | 7–12 | radians |
| Commanded left pose | `action` | 0–5 | radians |
| Commanded right pose | `action` | 7–12 | radians |

Indices 6 and 13 are grippers. An arbitrary 14-D robot vector is not accepted semantically just
because its shape matches: this implementation verifies both 14-D name lists from
`meta/info.json` before derivation. A future generic compiler should accept an explicit robot
schema instead of hard-coding ABC's layout.

When a dataset root is supplied, only `data/**/*.parquet` is read. LeRobot metadata Parquets under
`meta/` are deliberately excluded.

## Why this can generalize

The broader product primitive is not “fill nulls with a model.” It is a typed transformation with
a measurable invariant:

```text
known robot description + joint telemetry + named frame -> independently checkable geometry
```

The same compiler pattern can later cover URDF/MJCF kinematics, verified wrist-camera mounts, unit
and frame normalization, and format migration. Each transformation needs its own contract,
validation corpus, and refusal modes.
