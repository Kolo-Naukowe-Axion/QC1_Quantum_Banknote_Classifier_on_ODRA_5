# Finish incomplete depth-4 jobs, then run depth-6 in iteration 1 and 2.
#
# Queue order (single process, resume-safe):
#   1. iteration_2 / depth 4  — resume any incomplete (ansatz_odra + ansatz_simulator)
#   2. iteration_1 / depth 6  — both ansatzes
#   3. iteration_2 / depth 6  — both ansatzes
#
# Usage (from repository root, PowerShell):
#   $env:IQM_TOKEN = "..."
#   .\scripts\run_iqm_kl_finish_d4_then_d6.ps1
#
# Environment overrides:
#   RUN_ID, SAMPLES, SHOTS, N_BINS, SEED,
#   HARDWARE_RETRIES, RETRY_WAIT_SECONDS, RETRY_MAX_WAIT_SECONDS,
#   SKIP_QPU, SKIP_ANALYSIS, ANALYSIS_N_BINS

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
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

function Env-Default([string]$Name, [string]$Default) {
    if ([string]::IsNullOrEmpty((Get-Item "Env:$Name" -ErrorAction SilentlyContinue).Value)) {
        return $Default
    }
    return (Get-Item "Env:$Name").Value
}

function Invoke-KlStage {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$OutputDir,
        [Parameter(Mandatory = $true)][int[]]$Depths
    )

    Write-Host ""
    Write-Host "=== Stage: $Label ==="
    Write-Host "Output: $OutputDir"
    Write-Host "Depths: $($Depths -join ', ')"

    $args = @(
        "scripts/run_iqm_kl_expressibility.py",
        "--output-dir", $OutputDir,
        "--depth"
    ) + $Depths + @(
        "--samples", "$Samples",
        "--shots", "$Shots",
        "--n-bins", "$NBins",
        "--seed", "$Seed",
        "--iterations", "1",
        "--skip-iteration-precision",
        "--resume",
        "--hardware-retries", "$HardwareRetries",
        "--retry-wait-seconds", "$RetryWaitSeconds",
        "--retry-max-wait-seconds", "$RetryMaxWaitSeconds"
    )
    Invoke-ProjectPython -ArgumentList $args
}

$RunId = Env-Default "RUN_ID" "kl_hardware"
$Samples = [int](Env-Default "SAMPLES" "60")
$Shots = [int](Env-Default "SHOTS" "2048")
$NBins = [int](Env-Default "N_BINS" "400")
$Seed = [int](Env-Default "SEED" "42")
$HardwareRetries = [int](Env-Default "HARDWARE_RETRIES" "6")
$RetryWaitSeconds = [double](Env-Default "RETRY_WAIT_SECONDS" "60")
$RetryMaxWaitSeconds = [double](Env-Default "RETRY_MAX_WAIT_SECONDS" "600")
$SkipQpu = Env-Default "SKIP_QPU" "0"
$SkipAnalysis = Env-Default "SKIP_ANALYSIS" "1"
$AnalysisNBins = Env-Default "ANALYSIS_N_BINS" "75"

$Protocol = Join-Path $Root "evaluation_and_comparison/iqm_spark/kl_hardware_protocol.json"
$RunDir = Join-Path $Root "evaluation_and_comparison/iqm_spark/iqm_kl_outputs/$RunId"
$Iter1Dir = Join-Path $RunDir "iteration_1"
$Iter2Dir = Join-Path $RunDir "iteration_2"

Write-Host "=== KL queue: finish depth 4 (iter 2) -> depth 6 (iter 1) -> depth 6 (iter 2) ==="
Write-Host "Run ID:     $RunId"
Write-Host "Run output: $RunDir"
Write-Host "Samples:    $Samples"
Write-Host "Shots:      $Shots"
Write-Host "Seed:       $Seed"

if ($SkipQpu -ne "1" -and [string]::IsNullOrEmpty($env:IQM_TOKEN)) {
    Write-Error "Set `$env:IQM_TOKEN before running QPU stages."
    exit 1
}

if ($SkipQpu -ne "1") {
    Invoke-KlStage -Label "Finish depth 4 in iteration 2" -OutputDir $Iter2Dir -Depths @(4)
    Invoke-KlStage -Label "Depth 6 in iteration 1" -OutputDir $Iter1Dir -Depths @(6)
    Invoke-KlStage -Label "Depth 6 in iteration 2" -OutputDir $Iter2Dir -Depths @(6)
} else {
    Write-Host "[qpu] Skipped (SKIP_QPU=1)."
}

if ($SkipAnalysis -ne "1") {
    Write-Host ""
    Write-Host "=== Offline analysis ==="
    Invoke-ProjectPython @(
        "scripts/analyze_iqm_kl_hardware.py",
        "--run-dir", $RunDir,
        "--protocol-json", $Protocol,
        "--bootstrap-trials", "5000",
        "--confidence-levels", "0.90", "0.95",
        "--n-bins", "$AnalysisNBins"
    )
} else {
    Write-Host "[analysis] Skipped (SKIP_ANALYSIS=1)."
}

Write-Host ""
Write-Host "Done. Queue completed under $RunDir"
