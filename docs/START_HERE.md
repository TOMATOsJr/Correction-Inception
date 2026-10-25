# START HERE: running the LMA project on this PC

Everything is already in `C:\lma` (code, data, cache). Details and troubleshooting: `C:\lma\LMA Project\docs\LOCAL_RUN.md`.

## Requirements
* NVIDIA RTX 5070 Ti (16 GB), driver **570 or newer**. Check in PowerShell: `nvidia-smi`
* **60 GB free** on C:
* Internet (downloads ~28 GB once)
* Windows Settings → Power → **Sleep: Never**
* **Python 3.12** from python.org, with **"Add python.exe to PATH"** ticked
* Your **Hugging Face Read token** (account that accepted FERMAT's terms)

## Steps (PowerShell)
**1. Token** (replace with your token):
```powershell
Set-Content "C:\lma\LMA Project\.env" "HF_TOKEN=hf_your_token_here"
```

**2. Setup** (~1–1.5 h, mostly downloading). Must end with **SETUP OK**:
```powershell
cd "C:\lma\LMA Project"
powershell -ExecutionPolicy Bypass -File scripts\setup_local.ps1 -Data C:\lma
```

**3. Smoke test** (~15 min). Must end with **ALL DONE**; if not, STOP and send `results_smoke\logs\`:
```powershell
C:\lma\v\Scripts\python.exe scripts\run_all.py --data C:\lma --limit 3
Remove-Item -Recurse results_smoke, local_results_smoke.zip
```

**4. Full run** (unattended, ~7–11 h). Set `--hours` to the time you have left:
```powershell
C:\lma\v\Scripts\python.exe scripts\run_all.py --data C:\lma --hours 9
```
If it stops for any reason, run the same command again: it resumes.

**5. Finish**
* Upload `C:\lma\LMA Project\local_results.zip` to Drive.
* Delete `C:\lma`.
* Revoke the token at huggingface.co → Settings → Access Tokens.
