"""Pipeline stages, run in order: validate, ingest, audit, generate, label, featurize,
train, evaluate, register. Each stage is a `run(cfg)` function that reads the previous
stage's files and writes its own, so any stage can be rerun alone."""
