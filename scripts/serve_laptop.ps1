# Start the live guard on this laptop (CPU), with the real model and the trained detector.
#
# Run from the bluff-detector folder, in VS Code's terminal:
#     powershell -ExecutionPolicy Bypass -File scripts\serve_laptop.ps1                 # M0
#     powershell -ExecutionPolicy Bypass -File scripts\serve_laptop.ps1 -Variant m1_rag  # M1 (RAG)
#     powershell -ExecutionPolicy Bypass -File scripts\serve_laptop.ps1 -Variant m2_qlora  # M2
# Then open http://127.0.0.1:8000 in a browser. Stop it with Ctrl+C.
#
# Needs: the project environment (.venv) with torch and transformers (and peft for M2), the
# trained detector in artefacts/<variant>/serving/, and the pinned model's files in $ModelDir
# (about 3 GB). M1 also reads data/corpus/retrieval_chunks.jsonl; M2 loads the adapter in
# artefacts/adapters/m2_qlora/. Every question asked is logged to logs/requests.jsonl;
# `bluff monitor` turns that log into a drift report (reports/<variant>/monitoring/).

param(
    [ValidateSet("m0_base", "m1_rag", "m2_qlora")]
    [string]$Variant = "m0_base",
    [string]$ModelDir = "$env:USERPROFILE\models\qwen2.5-1.5b-instruct",
    [int]$Port = 8000
)

if (-not (Test-Path "$ModelDir\model.safetensors")) {
    Write-Error "No model at $ModelDir. Download the pinned Qwen2.5-1.5B-Instruct files there first."
    exit 1
}
if ($Variant -eq "m2_qlora" -and -not (Test-Path "artefacts\adapters\m2_qlora\adapter_model.safetensors")) {
    Write-Error "No M2 adapter in artefacts\adapters\m2_qlora. Pull the repo again (it is in git)."
    exit 1
}
$env:BLUFF_CONFIG = "configs/variants/$Variant.yaml"
$env:BLUFF_MODEL_DIR = $ModelDir
$env:BLUFF_DEVICE = "cpu"  # whole model in RAM (about 3 GB); "auto" offloads part to disk: slow
Write-Host "Loading $Variant on the CPU (about a minute), then open http://127.0.0.1:$Port"
& .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port $Port
