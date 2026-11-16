# Start the live guard on this laptop (CPU), with the real model and the trained detector.
#
# Run from the bluff-detector folder, in VS Code's terminal:
#     powershell -ExecutionPolicy Bypass -File scripts\serve_laptop.ps1
# Then open http://127.0.0.1:8000 in a browser. Stop it with Ctrl+C.
#
# Needs: the project environment (.venv) with torch and transformers, the trained detector in
# artefacts/<variant>/serving/, and the pinned model's files in $ModelDir (about 3 GB).
# Every question asked is logged to logs/requests.jsonl; `bluff monitor` turns that log into a
# drift report (reports/<variant>/monitoring/drift_report.md).

param(
    [string]$Config = "configs/variants/m0_base.yaml",
    [string]$ModelDir = "$env:USERPROFILE\models\qwen2.5-1.5b-instruct",
    [int]$Port = 8000
)

if (-not (Test-Path "$ModelDir\model.safetensors")) {
    Write-Error "No model at $ModelDir. Download the pinned Qwen2.5-1.5B-Instruct files there first."
    exit 1
}
$env:BLUFF_CONFIG = $Config
$env:BLUFF_MODEL_DIR = $ModelDir
$env:BLUFF_DEVICE = "cpu"  # whole model in RAM (about 3 GB); "auto" offloads part to disk: slow
Write-Host "Loading the model on the CPU (about a minute), then open http://127.0.0.1:$Port"
& .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port $Port
