# Running the correction pipeline on a local GPU (tested setup: Windows; card: RTX 5070 Ti, 16 GB)

The bundle contains student work. Keep it private: don't commit it, upload it, or share it.
Only `correction_metrics.zip`, made at the end of a run, is safe to send (numbers only).

## Is it GPU heavy?

Two steps need the GPU, and they use one model at a time:

| Step | Work | VRAM | Note |
|---|---|---|---|
| Token scores | 1 pass per page | ~8-12 GB (estimate) | ~10 s/page on a Kaggle T4; a 5070 Ti should be faster |
| Forced-choice check | 6-8 passes per checked number | ~8-12 GB (estimate) | the slow step; time grows with the numbers checked |
| Grading (`--grade`, optional) | 7B model, 4-bit | ~6-8 GB (estimate) | slow, only for the final comparison |

Everything else runs on the CPU in seconds or minutes. Pure what-if questions (`sweep_correct.py`) need no GPU.

Disk: ~25 GB free (models ~7 GB, ~15 GB more with `--grade`, data ~2 GB, results).

## 1. Install (once)

Needs Python 3.12 with "Add python.exe to PATH" ticked, and a current NVIDIA driver. Unzip the bundle to a **short
path** (long paths break the PyTorch install on Windows). Then in PowerShell, from the unzipped folder:

```powershell
powershell -ExecutionPolicy Bypass -File setup_correction.ps1
```

This creates `C:\lma\v`, installs everything, unzips the data to `C:\lma\data`, and checks that PyTorch sees the GPU.

## 2. Smoke test (a few minutes plus the 7 GB model download)

```powershell
C:\lma\v\Scripts\python.exe scripts\run_correction.py --data C:\lma\data --smoke
```

It must end with `send this file ...`. With 5 pages the numbers mean nothing; it only proves every step runs.
Note the `s/it` on the `verifying` bar: that times pages, so seconds per page x 293 estimates a topk3 run.

## 3. Real runs

Start with the cheap one, then read its result before spending more:

```powershell
C:\lma\v\Scripts\python.exe scripts\run_correction.py --data C:\lma\data --experiments topk3
```

Then, if useful:

```powershell
C:\lma\v\Scripts\python.exe scripts\run_correction.py --data C:\lma\data --experiments baseline
```

A crash or restart loses nothing: run the same command again. See all experiments with `--list`:

| Experiment | Question it answers | Relative cost |
|---|---|---|
| `topk3` | 3 most doubtful numbers per page: can we fix anything? | low |
| `baseline` | learned selector and all guards: the planned pipeline | high (~10 numbers/page) |
| `base_reader` | does the fine-tuned reader beat the untuned one? | high + its own scores |
| `crop` | does zooming around the number help? (position is estimated) | low |
| `more_cands` | does a longer candidate list raise candidate recall? | low-medium |
| `other` | Gemini's value as a candidate (flattering: experts corrected Gemini) | low |
| `gemini` | same pipeline on Gemini's transcripts (few misreads) | low + its own scores |

Combine with commas: `--experiments topk3,crop,more_cands`. Experiments on the same transcripts and reader reuse
scores. Add `--grade` only once a run reports more fixes than broken numbers.

## 4. Results

`results\correction\correction_metrics.zip`, and the printed summary. Read `summary.json` in this order:

1. `candidate_recall`: was the true number among the candidates? This is the ceiling for any fix.
2. `auto_fixed` vs `auto_broken_correct`: fixes must clearly beat breakages. If not, no automatic changes.
3. `auto_wrong_fix`: misreads changed to another wrong value.
4. `ta_per_page`: how much work goes to a TA.
5. `sweep_metrics.csv`: the same data under other rules (error budget, guards on/off, digit-doubt filter).

All numbers are on students whose thresholds were tuned on other students.

## Common problems

| Symptom | Fix |
|---|---|
| `no CUDA GPU` | reinstall PyTorch: `C:\lma\v\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` |
| `CUDA out of memory` | close games/browsers using the GPU and re-run; if it persists add `--max-pixels 600000` (use a new `--out`: it changes what the reader sees) |
| HF download slow or rate limited | create `.env` here with `HF_TOKEN=hf_...` (never share or commit it) |
| Anything else | send the last 30 lines of the output |
