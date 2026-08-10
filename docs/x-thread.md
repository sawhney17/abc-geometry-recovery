# X launch thread

Use the nine posts below as one thread. The bracketed visual directions are production notes, not
post copy.

Ready-to-post hero asset: [`abc-public-40-launch-card.png`](assets/abc-public-40-launch-card.png)
([editable SVG](assets/abc-public-40-launch-card.svg)).

## Post 1 — the finding

I audited all 40 public raw MCAPs in the Voxel51 ABC-130K preview.

314,081 of 641,593 arm records have `pose=[]` — 48.9533%.

Eight episodes are fully affected: every left/right state/action pose is empty.

The robot data is not necessarily gone, though. 🧵

[Visual 1: attach the ready-to-post launch card above.]

## Post 2 — why this slice is repairable

All 314,081 rows still carry six finite joint values.

So arm-local recovery is deterministic:

`6 joints + pinned YAM chain -> grasp-site SE(3)`

No learned imputation. No source mutation. If the frame is unprovable, the compiler refuses it.

[Visual 2: a real affected record with `pose: []` beside its six joint values, then the derived
arm-local pose and a green `FK valid` mask. Keep shared-frame output visibly locked.]

## Post 3 — the falsification test

“The math should work” was not enough.

I replayed the chain against 327,512 intact public poses. With the correct episode cohort:

- max position error: `6.67e-16 m`
- max rotation error: `4.52e-6°`

That is falsification against recorded data—not a plausible-looking fill.

[Visual 3: scrub an intact episode with recorded and FK trajectories overlaid. Show the residual
plot and a toggle that deliberately hides the recorded pose before reconstruction.]

## Post 4 — the hidden 19 cm trap

The full scan found something the four-episode canary missed:

- 30 intact episodes use right-base offset `[0, -0.61, 0] m`
- 2 use `[0, -0.80, 0] m`

Hardcoding `-0.61 m` silently puts 16,118 valid right-arm records off by exactly 19 cm.

[Visual 4: two sharp clusters at `-0.61 m` and `-0.80 m`, connected by a red `19 cm` bracket.]

## Post 5 — the honest repair boundary

Repair boundary: the eight missing-pose episodes cannot reveal their shared-frame cohort.

Arm-local pose is recoverable. Absolute right-arm pose needs rig calibration or a trusted cohort ID.

A tool that “fills every null” would manufacture geometry.

[Visual 5: the compiler emits `repaired: arm_local` and `refused: shared_frame / ambiguous base
offset` for the same row.]

## Post 6 — exact scope

Scope matters.

This is all 40 public raw MCAPs at pinned Voxel51 revision
`9659e8ce4b39580f48369cc31bc2e47a217c40e7`.

It is not confirmation of 60,982 affected episodes across the gated raw corpus. That number remains
attributed to amazon-far/abc#13.

## Post 7 — independent reproduction

Then I reran a separate field scanner:

- same 641,593 target records
- same 314,081 empty poses
- all six-joint vectors valid

Different code path, same result.

Robot-data claims should ship with pinned sources and counts anyone can recompute.

## Post 8 — what is actually shipped

This is the product: a deterministic compiler for robot logs.

`physics check -> safe derivation OR quarantine -> keyed sidecar + provenance`

Demo: https://sawhney17.github.io/abc-geometry-recovery/

Code: https://github.com/sawhney17/abc-geometry-recovery

[Visual 6: a four-stage pipeline. The source MCAP remains untouched while the sidecar and quarantine
report appear on separate branches.]

## Post 9 — call for a lab pilot

Release: https://github.com/sawhney17/abc-geometry-recovery/releases/tag/v0.1.0

Have logs blocked by missing fields or bad frames? Send 10 episodes plus the pinned robot model.

I will return a repair/quarantine report—including what the evidence cannot recover.

## Demo capture checklist

Capture one continuous 45–60 second demo that makes the claim inspectable:

1. Open the pinned 40-episode inventory and filter to the eight fully affected episodes.
2. Select one raw arm message and show the empty pose plus its intact six-joint telemetry.
3. Run arm-local FK, then scrub a separate intact episode with recorded and derived poses overlaid.
4. Switch to the cohort view and expose the `-0.61 m` and `-0.80 m` right-base clusters.
5. Attempt shared-frame recovery on a missing-pose episode and show the explicit refusal.
6. End on source hashes unchanged, a joinable sidecar, and a quarantine manifest.

Do not use a synthetic “before/after robot” animation without also showing the raw field, keyed
record, and refusal state. The interesting demo is the evidence trail.

## Claim discipline

- Say “all 40 public raw MCAPs at the pinned Voxel51 revision,” not “ABC-130K as a whole.”
- Say “314,081 arm-local FK-recoverable missing poses,” not “314,081 fully calibrated poses.”
- Say the eight episodes have empty pose fields across all four audited arm topics.
- Attribute the 60,982-episode corpus-wide figure to
  [amazon-far/abc#13](https://github.com/amazon-far/abc/issues/13); do not present it as reproduced.
- Treat the two right-base offsets as observed cohorts, not a universal station specification.
- Do not claim policy-quality or reward improvement without a controlled training ablation.
