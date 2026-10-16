PY ?= python
CONFIG ?= configs/variants/dev_fake.yaml

.PHONY: setup lock lint test pipeline serve docker smoke compare clean

setup:            ## install locked deps + the package in editable mode
	pip install --require-hashes -r requirements.lock
	pip install --no-deps -e .
	pip install -r requirements-dev.txt

lock:             ## recompile the hashed lock file after changing pyproject.toml
	uv pip compile pyproject.toml --python-version 3.11 --generate-hashes -o requirements.lock

lint:
	ruff check .

test:
	pytest -q

pipeline:         ## run every stage for CONFIG
	bluff all --config $(CONFIG)

serve:
	BLUFF_CONFIG=$(CONFIG) uvicorn app.main:app --reload --port 8000

docker:
	docker build -t bluff-detector:local .

smoke: docker
	./scripts/smoke.sh bluff-detector:local

compare:          ## H3: base vs fine-tuned
	bluff compare --config configs/variants/m0_base.yaml --other configs/variants/m2_qlora.yaml

clean:
	rm -rf data/interim/* data/processed/* artefacts/* reports/*
	touch data/interim/.gitkeep data/processed/.gitkeep artefacts/.gitkeep reports/.gitkeep
