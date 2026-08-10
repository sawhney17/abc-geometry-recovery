# Pilot: deterministically recover or quarantine 10 robot episodes

## The offer

Run one fixed-scope pilot on one robot and one declared collection configuration. The lab provides
10 representative episodes, including suspected failures and intact controls. The pilot classifies
every requested field as observed, deterministically recoverable, or refused, then emits repaired
values only in a joinable sidecar. Source logs are never overwritten.

This is not a generic anomaly dashboard and it is not learned gap filling. It is an evidence-gated
compiler pass:

```text
versioned telemetry + versioned physical model + named frame contract
  -> validated derivation OR explicit quarantine
```

The public ABC result demonstrates why the refusal path is part of the product. Across all 40 raw
MCAPs in the pinned Voxel51 preview, eight episodes are fully affected and 314,081 of 641,593 arm
records have empty poses (`48.9533%`). All retain valid six-joint telemetry, so arm-local FK is
available. The chain also validates against 327,512 intact records. Those controls reveal two
right-base cohorts, `-0.61 m` and `-0.80 m`; applying one transform universally creates a 19 cm error
on 16,118 records.

## Required pilot inputs

Before accepting data, agree in writing on:

- exactly 10 episodes from one robot family, with at least three intact control episodes and the
  suspected failures represented;
- immutable source identifiers or source-file hashes;
- the exact URDF/MJCF revision and any licensed meshes required to evaluate it;
- joint names, order, units, limits, and whether each stream is observed state or commanded action;
- the requested output frame and every supplied static transform or calibration revision;
- timestamp units, clocks, and the downstream join keys;
- the fields to audit and the validation tolerance for each deterministic derivation;
- where the data may be processed, retained, and deleted, and who is allowed to see it.

If one of those contracts is unknown, that uncertainty becomes a named pilot finding. It is not
silently guessed.

## Work performed

1. Fingerprint every source and inventory schemas, topics, rates, units, missingness, and finite-value
   violations.
2. Separate robot/rig cohorts before evaluating any cross-frame transform.
3. Withhold intact records, derive the requested field from the remaining allowed inputs, and compare
   against the recorded values.
4. Freeze the accepted model, frame contract, tolerance, and software environment.
5. Derive only records that satisfy that contract; issue a structured refusal code for every other
   record.
6. Re-read and re-hash all inputs after processing, then reconcile source, output, and quarantine
   counts.

## Deliverables

- an episode-by-field audit matrix with exact observed, recoverable, and refused counts;
- residual distributions and worst-case examples from the withheld intact controls;
- a keyed, append-only sidecar containing accepted derived values and validity masks;
- a quarantine manifest with record keys, reason codes, and the evidence needed to unblock each case;
- a provenance manifest covering source, model, calibration, schema, and tool revisions;
- a reproducible command or script plus a short engineering readout of the go/no-go result.

The stable public reference implementation and release are:

- https://github.com/sawhney17/abc-geometry-recovery
- https://github.com/sawhney17/abc-geometry-recovery/releases/tag/v0.1.0
- https://sawhney17.github.io/abc-geometry-recovery/

## Acceptance criteria

The pilot is accepted only when all applicable gates pass:

1. **Source preservation:** every source hash is identical before and after the run; source writes are
   zero.
2. **Accounting:** every in-scope episode and target record lands in exactly one terminal class:
   observed-valid, derived-valid, or refused. Totals reconcile with the decoder's raw topic counts.
3. **Join integrity:** every sidecar row has one unique source key; there are no duplicate, orphaned,
   or reordered identities.
4. **Held-out validation:** the derivation passes the tolerance agreed before processing on the intact
   controls. The report includes the full residual distribution and maximum, not only an average.
5. **Canary detection:** deliberately removed, malformed, and nonfinite test values are all detected
   and receive the expected repair or refusal class.
6. **Determinism:** a second clean run with pinned inputs produces identical keyed values, masks,
   classifications, and provenance.
7. **No silent fallback:** every missing prerequisite, model failure, cohort ambiguity, or tolerance
   failure creates a machine-readable refusal. It never produces a plausible default value.

If held-out validation fails, the valid deliverable is the audit and quarantine report—not a repaired
dataset.

## Safety and refusal boundaries

The pilot will not:

- infer a cross-arm, camera, world, or tool transform without validated calibration or an identified
  rig cohort;
- guess joint order, units, quaternion convention, timestamp units, or coordinate-frame semantics;
- use a learned model to fill a field described as deterministic;
- turn nonfinite, out-of-limit, stale, or temporally unmatched inputs into valid-looking outputs;
- infer task success, contact, force, synchronization, or camera calibration from weak proxies and
  label them as ground truth;
- modify source logs, publish lab data, or send it to a third-party model/API without explicit written
  authorization;
- operate a robot or claim the repaired data improves policy reward without a separate controlled
  evaluation.

The pilot refuses derivation when the model cannot be pinned, the joint contract is ambiguous, intact
controls are insufficient, multiple calibration cohorts cannot be identified, or the agreed residual
gate fails. Refused rows stay visible and joinable so an engineer can repair the missing prerequisite
later.

## Pilot scorecard

Report these outcomes without collapsing them into one “data quality” score:

- recording minutes and records audited;
- records already valid, deterministically recovered, and quarantined;
- recovery coverage by field, stream, episode, and rig cohort;
- withheld-control residual distribution and tolerance failures;
- count of failures caught before training;
- engineer decisions unblocked and recollection avoided;
- unresolved prerequisites, each with an owner and next action.

Do not use policy reward, training loss, or “hours made trainable” as a success claim unless that
downstream experiment was actually run.

## First outreach email

**Subject:** can 10 of [lab]'s failed robot logs be recovered safely?

Hey [name] — I noticed [specific collection stack, dataset, or failure mode].

I audited every raw MCAP in the public 40-episode Voxel51 preview of ABC-130K. Eight episodes were
missing every audited arm pose: 314,081 empty pose records in total. All retained valid joint
telemetry, so arm-local geometry was recoverable by deterministic FK. The same audit also caught two
different right-arm base offsets; hardcoding one would have introduced a 19 cm error.

I am offering a 10-episode pilot for one robot/configuration. You provide suspected failures, intact
controls, and the versioned robot/calibration files. I return a non-destructive sidecar for what passes
held-out validation and a keyed quarantine report for everything ambiguous.

Demo: https://sawhney17.github.io/abc-geometry-recovery/

Would it be useful to test this on one failed collection run?

## Short DM

I found 314,081 empty arm-pose records across all 40 public raw MCAPs in the Voxel51 ABC preview.
Their joints made arm-local recovery deterministic, but a hidden 19 cm rig-offset difference made
shared-frame filling unsafe. If you have 10 failed logs plus intact controls and the robot model, I can
return the same repair-or-quarantine evidence for your pipeline. Useful?

## Scope statement for every proposal

The independently reproduced finding covers all 40 public raw MCAPs at Voxel51 revision
`9659e8ce4b39580f48369cc31bc2e47a217c40e7`. It does not establish the prevalence of missing poses
across the gated 130k-episode raw corpus. The larger 60,982-episode figure remains a third-party claim
from [amazon-far/abc#13](https://github.com/amazon-far/abc/issues/13).
