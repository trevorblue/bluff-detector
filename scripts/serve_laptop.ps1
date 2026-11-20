# Start the live guard on this laptop (CPU), with the real model and the trained detectors.
#
# Run from the bluff-detector folder, in VS Code's terminal:
#     powershell -ExecutionPolicy Bypass -File scripts\serve_laptop.ps1
# Then open http://127.0.0.1:8000 in a browser. Stop it with Ctrl+C.
#
# By default it loads M0, M1, M2 and M3 together and the page lets you switch between them per
# question. They share ONE copy of the model in memory: M0 and M1 use the base model (M1 adds
# three retrieved passages), M2 switches the fine-tuning adapter on, M3 does both. M1 and M3
# take about 8 minutes per answer on a laptop CPU (long prompts). To load fewer:
#     ... serve_laptop.ps1 -Variants m0_base
#     ... serve_laptop.ps1 -Variants m0_base,m2_qlora -Default m2_qlora
#
# Needs: the project environment (.venv) with torch, transformers and peft, the trained
# detectors in artefacts/<variant>/, the M2 adapter in artefacts/adapters/m2_qlora/ (in git; M3
# uses it too), data/corpus/retrieval_chunks.jsonl for M1 and M3, and the pinned model's files in $ModelDir (about
# 3 GB). Every question asked is logged to logs/requests.jsonl; `bluff monitor` turns that log
# into a drift report (reports/<variant>/monitoring/).

param(
    [string[]]$Variants = @("m0_base", "m1_rag", "m2_qlora", "m3_rag_qlora"),
    [ValidateSet("m0_base", "m1_rag", "m2_qlora", "m3_rag_qlora")]
    [string]$Default = "m0_base",
    [string]$ModelDir = "$env:USERPROFILE\models\qwen2.5-1.5b-instruct",
    [int]$Port = 8000
)

if (-not (Test-Path "$ModelDir\model.safetensors")) {
    Write-Error "No model at $ModelDir. Download the pinned Qwen2.5-1.5B-Instruct files there first."
    exit 1
}
if (($Variants -contains "m2_qlora" -or $Variants -contains "m3_rag_qlora") -and -not (Test-Path "artefacts\adapters\m2_qlora\adapter_model.safetensors")) {
    Write-Error "No M2 adapter in artefacts\adapters\m2_qlora. Pull the repo again (it is in git)."
    exit 1
}
if ($Variants -notcontains $Default) { $Default = $Variants[0] }
$env:BLUFF_CONFIG = "configs/variants/$Default.yaml"
$env:BLUFF_VARIANTS = ($Variants -join ",")
$env:BLUFF_MODEL_DIR = $ModelDir
$env:BLUFF_DEVICE = "cpu"  # whole model in RAM (about 3 GB); "auto" offloads part to disk: slow
Write-Host "Loading $($Variants -join ', ') on the CPU (about a minute), then open http://127.0.0.1:$Port"
& .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port $Port
