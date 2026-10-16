#!/usr/bin/env bash
# Smoke test a built image: start it, wait for /ready, ask one question, check the contract.
set -euo pipefail
IMAGE="${1:?usage: smoke.sh <image>}"
NAME="bluff-smoke-$$"
docker run -d --rm --name "$NAME" -p 18000:8000 --read-only --tmpfs /tmp "$IMAGE" >/dev/null
trap 'docker stop "$NAME" >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 30); do
  if curl -fsS localhost:18000/ready >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS localhost:18000/ready
BODY=$(curl -fsS -X POST localhost:18000/v1/ask -H 'Content-Type: application/json' \
  -d '{"question": "How many counties does Kenya have?"}')
echo "$BODY"
echo "$BODY" | python3 -c "import json,sys; r=json.load(sys.stdin); assert r['verdict'] in {'answer','answer_with_citation','abstain'}; assert 0 <= r['bluff_score'] <= 1"
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST localhost:18000/v1/ask \
  -H 'Content-Type: application/json' -d '{"question": ""}')
[ "$CODE" = "422" ] || { echo "expected 422 for an empty question, got $CODE"; exit 1; }
echo "smoke test passed"
