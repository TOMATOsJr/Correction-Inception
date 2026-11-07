# One-time setup for the correction pipeline, run from the unzipped bundle (docs\CORRECTION_ON_LOCAL_GPU.md):
#   powershell -ExecutionPolicy Bypass -File setup_correction.ps1
# Python environment at <Base>\v (short path: PyTorch fails to install under long paths), data at <Base>\data.
param([string]$Base = "C:\lma", [string]$Cuda = "cu128")
$ErrorActionPreference = "Continue"  # native tools write progress to stderr; we check exit codes instead
$Root = $PSScriptRoot
$Venv = Join-Path $Base "v"
$Py = Join-Path $Venv "Scripts\python.exe"
$Data = Join-Path $Base "data"

function Run($what, [scriptblock]$cmd) {
    Write-Host "`n=== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $what (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python not found: install Python 3.12 with 'Add python.exe to PATH' ticked" -ForegroundColor Red; exit 1
}
Run "python version" { python --version }
Run "nvidia driver" { nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv }
if (-not (Test-Path $Py)) { Run "create venv at $Venv" { python -m venv $Venv } }
Run "upgrade pip" { & $Py -m pip install -q --upgrade pip }
Run "PyTorch ($Cuda build)" { & $Py -m pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda" }
Run "pinned libraries" { & $Py -m pip install -q -r (Join-Path $Root "requirements-gpu.txt") }
Run "torch sees the GPU" { & $Py -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.__version__, torch.cuda.get_device_name(0))" }

foreach ($z in Get-ChildItem (Join-Path $Root "data_zips") -Filter *.zip) {
    $dst = Join-Path $Data $z.BaseName
    if (-not (Test-Path $dst)) {
        Write-Host "`n=== unzip $($z.Name)" -ForegroundColor Cyan
        Run "unzip $($z.Name)" { & $Py -m zipfile -e $z.FullName $dst }
    }
}
Write-Host "`nSETUP OK. Next, the smoke test:" -ForegroundColor Green
Write-Host "  $Py scripts\run_correction.py --data $Data --smoke"
