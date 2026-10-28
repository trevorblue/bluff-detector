"""The test suite. Run it with `make test` (or `pytest -q`); CI runs it on every push.

Four layers, fastest first: unit/ (one function at a time), data/ (the Kenya question set's
rules), api/ (the FastAPI service) and integration/ (the whole pipeline on the fake model).
CRISP-DM: evaluation of the code itself, and the gate before deployment.
"""
