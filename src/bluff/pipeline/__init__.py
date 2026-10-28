"""Pipeline stages, run in order: validate, ingest, audit, generate, label, featurize,
train, evaluate, register. Each stage is a `run(cfg)` function that reads the previous
stage's files and writes its own, so any stage can be rerun alone.

CRISP-DM: validate is data understanding; ingest, audit, generate, label and featurize are
data preparation; train is modeling; evaluate (and compare) is evaluation; register is
deployment. docs/architecture.md has the full map.
"""
