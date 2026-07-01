# Fixed-budget KL expressibility on IQM Spark (depths 2 and 4 only).
# See evaluation_and_comparison/iqm_spark/iqm_kl_hardware_methodology.html
#
# Usage (from repository root, PowerShell):
#   $env:IQM_TOKEN = "..."
#   .\scripts\run_iqm_kl_hardware_study.ps1
#
# Resume after interruption (default; per-sample CSV cache):
#   .\scripts\run_iqm_kl_hardware_study.ps1
#
# Second day on a separate run-id:
#   $env:RUN_ID = "kl_hardware_day2"
#   $env:ITERATIONS = "1"
#   $env:SKIP_BINS = "1"
#   .\scripts\run_iqm_kl_hardware_study.ps1
#
# Environment overrides (same names as the .sh script):
#   RUN_ID, DEPTHS, SAMPLES, SHOTS, N_BINS, SEED, ITERATIONS,
#   HARDWARE_RETRIES, RETRY_WAIT_SECONDS, RETRY_MAX_WAIT_SECONDS,
#   COMPARE_RUN_DIR, SKIP_BINS, SKIP_QPU, SKIP_ANALYSIS

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$UseUv = $null -ne (Get-Command uv -ErrorAction SilentlyContinue)
if (-not $UseUv) {
    Write-Warning "uv not found; falling back to python"
}

function Invoke-ProjectPython {
    param([Parameter(Mandatory = $true)][string[]]$ArgumentList)

    if ($UseUv) {
        & uv run python @ArgumentList
    } else {
        & python @ArgumentList
    }
}

function Env-Default([string]$Name, [string]$Default) {
    if ([string]::IsNullOrEmpty((Get-Item "Env:$Name" -ErrorAction SilentlyContinue).Value)) {
        return $Default
    }
    return (Get-Item "Env:$Name").Value
}

$RunId = Env-Default "RUN_ID" "kl_hardware"
$Depths = Env-Default "DEPTHS" "2 4"
$Samples = [int](Env-Default "SAMPLES" "60")
$Shots = [int](Env-Default "SHOTS" "2048")
$NBins = [int](Env-Default "N_BINS" "400")
$Seed = [int](Env-Default "SEED" "42")
$Iterations = [int](Env-Default "ITERATIONS" "2")
$HardwareRetries = [int](Env-Default "HARDWARE_RETRIES" "6")
$RetryWaitSeconds = [double](Env-Default "RETRY_WAIT_SECONDS" "60")
$RetryMaxWaitSeconds = [double](Env-Default "RETRY_MAX_WAIT_SECONDS" "600")
$SkipBins = Env-Default "SKIP_BINS" "1"
$SkipQpu = Env-Default "SKIP_QPU" "0"
$SkipAnalysis = Env-Default "SKIP_ANALYSIS" "0"
$CompareRunDir = Env-Default "COMPARE_RUN_DIR" ""

$Protocol = Join-Path $Root "evaluation_and_comparison/iqm_spark/kl_hardware_protocol.json"
$RunDir = Join-Path $Root "evaluation_and_comparison/iqm_spark/iqm_kl_outputs/$RunId"

$DepthArr = $Depths -split "\s+" | Where-Object { $_ -ne "" }
$Pairs = 2 * $DepthArr.Count * $Samples * $Iterations
$EstHours = [math]::Round($Pairs * $Shots / 4096 / 15, 1)

$PythonLabel = if ($UseUv) { "uv run python" } else { "python" }

Write-Host "=== KL hardware study (fixed budget) ==="
Write-Host "Python:       $PythonLabel"
Write-Host "Run ID:       $RunId"
Write-Host "Run output:   $RunDir"
Write-Host "Depths:       $Depths"
Write-Host "Samples/job:  $Samples"
Write-Host "Shots:        $Shots"
Write-Host "Iterations:   $Iterations"
Write-Host "Retries:      $HardwareRetries (wait ${RetryWaitSeconds}s, max ${RetryMaxWaitSeconds}s)"
Write-Host "Est. pairs:   $Pairs  (~$EstHours h QPU @ ~4 min/pair @ 4096 shots, scaled by S)"
Write-Host ""

if ($SkipQpu -ne "1" -and [string]::IsNullOrEmpty($env:IQM_TOKEN)) {
    Write-Error "Set `$env:IQM_TOKEN before running QPU stage."
    exit 1
}

if ($SkipBins -ne "1") {
    Write-Host "[bins] Offline bin sensitivity (no QPU)..."
    $binScript = @'
from pathlib import Path
import sys

ROOT = Path(".")
sys.path.insert(0, str(ROOT / "src"))
from qbanknote.metrics import choose_kl_bins, compute_kl_bin_sensitivity

_, aggregate = compute_kl_bin_sensitivity(
    num_qubits=5,
    n_samples=60,
    bin_grid=[50, 75, 100, 150, 200, 250, 300, 400],
    n_reference_bins=400,
    n_trials=100,
    seed=42,
)
chosen = choose_kl_bins(aggregate, tolerance=0.01)
print(chosen)
'@
    $NBins = [int](Invoke-ProjectPython @("-c", $binScript) | Select-Object -Last 1)
    Write-Host "[bins] Using n_bins=$NBins"
} else {
    Write-Host "[bins] Skipped (SKIP_BINS=1); using N_BINS=$NBins"
}

if ($SkipQpu -ne "1") {
    Write-Host "[qpu] Starting hardware sweep (resume + per-sample cache enabled)..."
    $qpuArgs = @(
        "scripts/run_iqm_kl_expressibility.py",
        "--run-id", $RunId,
        "--depth") + $DepthArr + @(
        "--samples", "$Samples",
        "--shots", "$Shots",
        "--n-bins", "$NBins",
        "--seed", "$Seed",
        "--iterations", "$Iterations",
        "--skip-iteration-precision",
        "--resume",
        "--hardware-retries", "$HardwareRetries",
        "--retry-wait-seconds", "$RetryWaitSeconds",
        "--retry-max-wait-seconds", "$RetryMaxWaitSeconds"
    )
    Invoke-ProjectPython -ArgumentList $qpuArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host "[qpu] Skipped (SKIP_QPU=1)."
}

if ($SkipAnalysis -ne "1") {
    Write-Host "[analysis] Offline bootstrap + drift + sim/Haar comparison..."
    $analyzeArgs = @(
        "scripts/analyze_iqm_kl_hardware.py",
        "--run-dir", $RunDir,
        "--protocol-json", $Protocol,
        "--bootstrap-trials", "5000",
        "--confidence-levels", "0.90", "0.95",
        "--n-bins", "$NBins"
    )
    if (-not [string]::IsNullOrEmpty($CompareRunDir)) {
        $analyzeArgs += @("--compare-run-dir", $CompareRunDir)
    }
    Invoke-ProjectPython -ArgumentList $analyzeArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host "[analysis] Skipped (SKIP_ANALYSIS=1)."
}

Write-Host ""
Write-Host "Done. Outputs under $RunDir/analysis/"
