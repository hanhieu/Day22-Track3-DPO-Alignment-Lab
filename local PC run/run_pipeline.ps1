# Day 22 DPO Lab — Pipeline runner for Windows
# Sets required env vars and runs all 6 notebooks in order

$env:COMPUTE_TIER = "T4"
$env:CUDA_PATH = "C:\Users\Admin\AppData\Roaming\Python\Python313\site-packages\triton\backends\nvidia"
$env:CC = "C:\Users\Admin\AppData\Roaming\Python\Python313\site-packages\triton\runtime\tcc\tcc.exe"

Write-Host "=== Day 22 DPO Lab Pipeline ===" -ForegroundColor Cyan
Write-Host "COMPUTE_TIER: $env:COMPUTE_TIER"
Write-Host "CUDA_PATH: $env:CUDA_PATH"
Write-Host "CC: $env:CC"
Write-Host ""

$notebooks = @(
    "notebooks/01_sft_mini.py",
    "notebooks/02_preference_data.py",
    "notebooks/03_dpo_train.py",
    "notebooks/04_compare_and_eval.py",
    "notebooks/05_merge_deploy_gguf.py",
    "notebooks/06_benchmark.py"
)

foreach ($nb in $notebooks) {
    Write-Host "=== Running $nb ===" -ForegroundColor Yellow
    $start = Get-Date
    python run_nb.py $nb
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: $nb failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    $elapsed = (Get-Date) - $start
    Write-Host "=== $nb completed in $($elapsed.ToString('mm\:ss')) ===" -ForegroundColor Green
    Write-Host ""
}

Write-Host "=== Pipeline complete! Run verify.py to check submission readiness ===" -ForegroundColor Cyan
python run_nb.py scripts/verify.py
