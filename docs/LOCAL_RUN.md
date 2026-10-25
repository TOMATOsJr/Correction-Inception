# Running everything on a local GPU (RTX 5070 Ti, 16 GB), in one visit

Two scripts do everything:
* `scripts\setup_local.ps1`: installs, checks and downloads (one time);
* `scripts\run_all.py`: the whole GPU job (Kaggle notebooks 01 + 02).

## Before the visit
1. Upload **`D:\lma\FOR_DRIVE\lma_bundle.zip`** (~600 MB: code + data + cache) to a **private** Drive folder.
   Share it only with the friend.
2. Have your **Hugging Face Read token** ready, from the account that accepted FERMAT's terms.
3. Optional: `notebooks/05_smoke_test_run_all_kaggle.ipynb` runs the same smoke test on Kaggle, if you want an extra check.

## Built-in safety nets
* **Preflight** (2–5 min, during setup): checks libraries, GPU, data, disk, token and FERMAT access, and downloads everything.
  Any blocking problem stops it *before* the long run.
* **4-bit fallback:** if the 4-bit library fails on this GPU, training automatically re-runs as plain LoRA
  (a 3B model fits 16 GB), so the key result still comes out. Only the grade-impact scoring needs 4-bit;
  if it fails, it is redone later on Kaggle from the saved transcripts.

## Timeline of the visit
| Phase | You at the PC? | Time |
|---|---|---|
| A. Install + checks + downloads (`setup_local.ps1`) | yes | 45–90 min (mostly downloading ~28 GB) |
| B. Smoke test (`--limit 3`) | yes, watch it | ~15 min |
| C. Full run (`run_all.py`) | **no, it runs alone** | ~7–11 h (estimate) |
| D. Upload `local_results.zip` + clean up | yes | 10 min |

If the visit is shorter than A+B+C, pass `--hours`. Exam reading then stops early, so everything finishes in time (see below).

## Phase A: setup
1. **Check the PC:**
   * `nvidia-smi` shows the 5070 Ti, with driver **570+**;
   * **≥ 60 GB free** on C:;
   * Settings → Power → **Sleep: Never**.
2. **Install Python 3.12** from python.org, ticking **"Add python.exe to PATH"**.
3. Download **`lma_bundle.zip`** from Drive and unzip it to **`C:\`** (right-click → Extract All → `C:\`).
   * You should get `C:\lma\LMA Project`, `C:\lma\edu_text` and `C:\lma\images`.
   * Keep this short path: Windows breaks long ones.
4. (Nothing to unzip separately: the code, data and grade cache are already in place.)
5. Create the token file:
   ```powershell
   Set-Content "C:\lma\LMA Project\.env" "HF_TOKEN=hf_your_token_here"
   ```
6. Run setup:
   ```powershell
   cd "C:\lma\LMA Project"
   powershell -ExecutionPolicy Bypass -File scripts\setup_local.ps1 -Data C:\lma
   ```
   * It must end with **SETUP OK** and **PREFLIGHT PASSED**.
   * Every line marked `FAIL` names its fix.
   * If PyTorch has no wheel for the GPU, re-run with `-Cuda cu129` (or `cu130`).

## Phase B: smoke test (catches anything left, in ~15 min)
```powershell
C:\lma\v\Scripts\python.exe scripts\run_all.py --data C:\lma --limit 3
```
* It must end with **ALL DONE**.
* If not, the failed step's log is in `results_smoke\logs\`. Send that file. Don't start phase C.
* When it passes, delete the smoke output:
  ```powershell
  Remove-Item -Recurse results_smoke, local_results_smoke.zip
  ```

## Phase C: full run
```powershell
C:\lma\v\Scripts\python.exe scripts\run_all.py --data C:\lma
```
With a time limit, e.g. 9 hours of access left:
```powershell
C:\lma\v\Scripts\python.exe scripts\run_all.py --data C:\lma --hours 9
```

**Order** (most important first):
1. train the faithful reader;
2. FERMAT test, fine-tuned (**silent corrections after**);
3. FERMAT test, untuned (**before**);
4. read the 293 exams (fine-tuned, then untuned);
5. score both.

**Safety nets:**
* a failing step doesn't stop the others; steps that need it are skipped;
* re-running the same command **resumes** (finished steps and pages are skipped);
* `--hours` stops exam reading early and keeps ~75 min for scoring and zipping. Partial exam sets are still scored;
* everything is logged in `results\logs\` (`run_all.log` + one log per step).

## Phase D: finish
* Upload **`C:\lma\LMA Project\local_results.zip`** (made automatically: results, logs, transcripts, trained adapter) to Drive.
* The last line says **ALL DONE**, or lists the steps with problems. Either way, upload the zip: the logs are inside.
* **Clean up:**
  * delete `C:\lma` (student data, models, token, Python env);
  * revoke the HF token at huggingface.co → Settings → Access Tokens.

## What's on Git vs Drive
| Item | Where |
|---|---|
| Code | private Git repo at the end (now: `lma_project_code.zip` on Drive) |
| `.env` / tokens | nowhere: typed on the PC, deleted after |
| `edu_text.zip`, images zip, `llm_cache.sqlite` | private Drive folder only (student data) |
| Models, FERMAT | downloaded by the setup script |
| Results | `local_results.zip` → Drive → laptop |
