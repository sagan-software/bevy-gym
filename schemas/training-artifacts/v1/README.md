# Training artifact schemas v1

This directory freezes the version `1.0.0` JSON contract for one `bevy-gym` training run and its
proof/video evidence. All schemas identify JSON Schema Draft 2020-12, reject unknown fields at
named object boundaries, and use lowercase SHA-256 strings plus canonical UUIDs for joins.

## Public schemas

- `config.schema.json`: immutable run identity, exact command/configuration, budget, qualification
  constraints, checkpoint schedule, and output paths.
- `seeds.schema.json`: precision-safe root/derived seeds, per-environment reset streams, fixed
  validation/test/demo suites, and explicit separation/application evidence.
- `metrics-row.schema.json`: one self-identifying `metrics.jsonl` row with elapsed time, complete
  step/episode/update counts, policy hash, and optional checkpoint link.
- `eval-row.schema.json`: one `eval.jsonl` row binding a fixed suite and evaluation process to exact
  checkpoint bytes, raw counts, statistics, confidence interval, and threshold result.
- `provenance.schema.json`: source/oracle commits, invocation, runtime/host, timing, counts, stop
  reason, hashed inputs, best checkpoint, and reproducibility evidence.
- `summary.schema.json`: concise initial-validation, best-validation, and held-out-test outcome.
- `proof.schema.json`: qualifying-run receipt with fixed validation selection, disjoint
  fresh-process test evaluation, exact checkpoint hash, threshold/confidence/raw counts, and an
  optional five-seed release cohort.
- `video-manifest.schema.json`: 30-second H.264/yuv420p output, poster, official viewport/cadence,
  and frame-accurate segments joined to original checkpoint steps, hashes, metrics/eval rows, and
  one fixed demo suite.

`common.schema.json` contains shared definitions. It is support infrastructure, not a run artifact.
Each line of a JSONL file is validated independently against its row schema.

## Validation

Run:

```sh
python3 schemas/training-artifacts/v1/validate.py
```

The standard-library-only validator:

1. parses every schema while rejecting duplicate JSON keys;
2. checks the Draft 2020-12 keyword shapes used here;
3. rejects unsupported/misspelled keywords and open named-object schemas;
4. resolves every local `$ref` and JSON Pointer;
5. validates one complete canonical instance for every public schema;
6. proves focused invalid instances fail; and
7. enforces cross-field `x-bevy-gym-invariants` that ordinary JSON Schema cannot express, including
   seed-suite disjointness, raw-count totals, collision policy, checkpoint/count/hash joins,
   fresh-process test evaluation, and contiguous video segments.

The host currently has no Draft 2020-12 meta-schema validator installed, and this subtree adds no
external dependency. Consequently, `validate.py` is intentionally limited to the vocabulary used
here and is not a general JSON Schema implementation. Validation against the official Draft
2020-12 meta-schema must be added when the project deliberately chooses a maintained validator.

## Versioning

Runtime writers should emit exactly `schema_version: "1.0.0"`. Breaking changes require a new
versioned directory and `$id`; do not silently reinterpret existing proof artifacts. Cross-file
runtime integration must verify hashes against bytes on disk rather than trusting self-reported
fields.
