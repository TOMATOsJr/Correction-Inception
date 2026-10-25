# One-time setup on the GPU PC. Run from the project folder in PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_local.ps1 -Data C:\lma
# Creates the Python environment at <Data>\v (SHORT path: PyTorch fails to install under long paths on Windows),
# installs the pinned libraries, then runs the preflight checks (which also download models + FERMAT).
param([string]$Data = "C:\lma", [string]$Cuda = "cu128")
$ErrorActionPreference = "Continue"  # native tools write progress to stderr; we check exit codes instead
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Data "v"
$Py = Join-Path $Venv "Scripts\python.exe"

function Run($what, [scriptblock]$cmd) {
    Write-Host "`n=== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $what (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
}

$sys = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $sys) { Write-Host "Python not found: install Python 3.12 with 'Add python.exe to PATH' ticked" -ForegroundColor Red; exit 1 }
Run "python version" { python --version }
Run "nvidia driver" { nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv }

if (-not (Test-Path $Py)) { Run "create venv at $Venv" { python -m venv $Venv } }
Run "upgrade pip" { & $Py -m pip install -q --upgrade pip }
Run "PyTorch ($Cuda build)" { & $Py -m pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda" }
Run "pinned libraries" { & $Py -m pip install -q -r (Join-Path $Root "requirements-gpu.txt") }
Run "torch sees the GPU" { & $Py -c "import torch, torchvision; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0))" }

if (-not (Test-Path (Join-Path $Root ".env"))) {
    Write-Host "`nNo .env yet. Create $Root\.env with one line:  HF_TOKEN=hf_..." -ForegroundColor Yellow
    exit 1
}
Run "preflight + downloads (models ~23 GB, FERMAT ~5 GB)" { & $Py (Join-Path $Root "scripts\run_all.py") --data $Data --preflight }
Write-Host "`nSETUP OK. Next: smoke test, then the full run (docs\LOCAL_RUN.md)." -ForegroundColor Green
