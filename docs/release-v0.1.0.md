# v0.1.0 - ABC public-40 missing-pose recovery

This release audits every raw MCAP in the ungated 40-episode Voxel51 ABC-130K preview and publishes
the first recovery slice produced from the observed failures.

## Primary result

- 40 revision-pinned source files, 3,509,376,768 bytes
- 641,593 exact arm state/action records
- 314,081 records with an empty protobuf pose field (48.9533%)
- 8 episodes with every target pose missing; 32 entirely intact; no mixed episodes
- all 314,081 missing-pose records retain six finite joints and are recovered arm-locally
- an independent literal field scanner reproduces every count and source hash

The intact cohort also exposes two right-base offsets: 30 episodes at `-0.61 m` and two at
`-0.80 m`. Hard-coding the common offset creates a false 19 cm error on 16,118 valid records. The
recovery artifact therefore emits arm-local geometry and refuses an unproven shared-frame
transform.

## Release assets

- `abc-130k-public-missing-pose-recovery.parquet`: 314,081 recovered rows, 17,742,217 bytes,
  SHA-256 `511bc785df6df8a080e97530f991669f3527444a2bcfddb4816846f0bb97bcb6`
- `abc-130k-public-missing-pose-recovery.manifest.json`: per-source expected/observed identity,
  status counts, model digest, schema, coordinate frame, and output digest
- `abc-130k-public-40-geometry-audit.json`: complete primary audit
- `abc-130k-independent-protobuf-field-scan.json`: independent field-length/finiteness cross-check
- `abc-130k-public-40-sources.tsv`: exact pinned public inventory
- `abc-geometry-file-000.parquet`: secondary 384,567-row LeRobot augmentation
- `release-v0.1.0.sha256`: checksums for every data/evidence asset

Two independent recovery builds produced byte-identical Parquet. The tool verifies every expected
source hash and byte size before decoding, rehashes after derivation, never modifies the MCAPs, and
publishes the manifest last as a commit marker.

## Scope boundary

This independently establishes the result for all 40 public MCAPs at Voxel51 revision
`9659e8ce4b39580f48369cc31bc2e47a217c40e7`. It does not establish the prevalence of missing poses
across the gated full corpus. The 60,982-episode corpus-wide figure remains attributed to
amazon-far/abc issue #13.

See the interactive proof and calibration-failure toggle at
https://sawhney17.github.io/abc-geometry-recovery/.

The corresponding opt-in exporter change is proposed upstream in
https://github.com/amazon-far/abc/pull/17.
