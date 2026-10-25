# One-time setup (PowerShell). Run from the project root:  .\scripts\setup.ps1
$ErrorActionPreference = "Stop"
if (-not (Test-Path "external\CIRCUIT_EDU_HW_ACL")) {
    git clone https://github.com/gt-learning-innovation/CIRCUIT_EDU_HW_ACL external\CIRCUIT_EDU_HW_ACL
}
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
if (-not (Test-Path ".env")) { "GEMINI_API_KEY=" | Out-File -Encoding utf8 .env; Write-Host "Created .env: add your key." }
Write-Host "Now download EDU-CIRCUIT-HW (see data\README.md) and unzip it into external\CIRCUIT_EDU_HW_ACL"
python src\eval\evaluate_recognition.py --self-test
