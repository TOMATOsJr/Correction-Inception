# Fine-tuning the A/B checker (branch `ab_checker`)

## Why
The Advocate and the Auditor rely on one tool more than any other: the forced-choice question
*"is the number marked [?] written as A) 59 or B) 54? Which did the student actually write, even if it is mathematically
wrong?"* (`src/detect/verify_choice.py`). In the correction run the untuned reader answered it at chance:

- when the true value was one of the options, it picked it only **45%** of the time;
- it preferred a wrong alternative for **23%** of correct numbers;
- so automatic changes fixed 2 misreads and broke 7 correct numbers.

No arbiter rule can turn a coin flip into net-positive corrections. This branch **fine-tunes the checker for exactly this
question**, so it answers from the ink instead of from what makes the maths right. If it gets well above 50% on
misreads while keeping correct numbers, the corrections, and then the agents that use them, can beat "no correction".

This is the fine-tuning part of the project feeding the tool-calling and agent parts: the same trained checker is the
Advocate's misread test and the Auditor's silent-correction test.

## Training pairs (`src/finetune/ab_data.py`)
Every pair is asked in both option orders, and the right answer is always **what is written on the page**.

| Kind | Source | Options | Right answer | Teaches |
|---|---|---|---|---|
| `silent` | FERMAT (planted errors) | student's wrong value vs the correct value | the wrong, written one | not to silently correct (Auditor) |
| `lookalike` | FERMAT and EDU | written number vs a look-alike digit swap (3↔8, 1↔7, …) | the written one | to keep correct numbers; stops "always pick the wrong-maths option" |
| `misread` | EDU (3 students) | transcript's value vs the expert's value, near misses only (one digit changed / dropped / added) | the expert's value | to fix real misreads (Advocate) |

- EDU "misreads" that are entirely different numbers are left out: most are alignment noise (`MISREAD_WORK_SUMMARY`, 5.6).
  That leaves 403 of 680.
- The 3 EDU training students are the ones with the most near misses (`student_2`, `student_35`, `student_42` on the
  current labels: 207 near misses). **They are left out of every evaluation**, so all reported numbers are on unseen
  students (196 near misses). Caveat: these are the messiest students, so the held-out set is somewhat cleaner.
- FERMAT is split by problem (`fermat_data.split_of`); evaluation uses FERMAT test problems only.
- The FERMAT and EDU pair counts are capped and shared out evenly (`train_checker.balance`), so the plentiful FERMAT
  look-alike pairs don't swamp the scarce EDU misread pairs.

## Two checkers
Both continue training our verbatim reader's LoRA (`--init-adapter`), at the verifier's resolution (602,112 pixels):

- `checker_f`: FERMAT pairs only (no EDU data at all);
- `checker_fe`: FERMAT + the 3 EDU students.

Loss is on a single token, the answer letter, which is exactly the token the verifier reads. The question is built by
`src/detect/choice_prompt.encode`, shared by training, evaluation and the verifier.

## Running on ADA
`job.sh` (repo root) is the SLURM job. It uses the same layout as the last correction run: data, transcripts and the
reader LoRA are found inside the repo (`LMA_DATA=.`), the venv is `/home2/prabhavsai.kollipara/venv`, and the models
come from the usual Hugging Face cache. The doubtful numbers come from the 600k-pixel scores of that run
(`results/correction_600k/qwen_ftreader/`), so the checker is tested on the same numbers as before.
`HF_TOKEN` must be in `.env`, from an account that accepted FERMAT's terms (FERMAT downloads once, ~5 GB).

```bash
cd /home2/prabhavsai.kollipara/Correction-Inception
git fetch origin && git checkout ab_checker && git pull

# 1. smoke test: every stage on a few examples (~20-30 min plus downloads). Read the end of output.txt.
LMA_SMOKE=1 sbatch job.sh
tail -f output.txt
cat results/checker_smoke/CHECKER_RESULTS.md     # tables filled with (meaningless) numbers = plumbing works
rm -rf results/checker_smoke

# 2. the real run
sbatch job.sh
cat results/checker/CHECKER_RESULTS.md
```

**Resuming:** every stage skips finished work, so after a crash or the time limit, submit the same command again.
To run only some stages: `STAGES="eval correct" sbatch job.sh`.

| Stage | What | GPU | Rough time (2× 2080 Ti) |
|---|---|---|---|
| `test` | unit tests | – | seconds |
| `pairs` | FERMAT download (once, ~5 GB) + pairs; EDU pairs | – | 10–20 min |
| `train` | `checker_f` on GPU 0 and `checker_fe` on GPU 1 at once | 2 | ~1–3 h (6,000 examples each) |
| `eval` | base, reader, `checker_f`, `checker_fe` on up to 400 pairs per kind | 2 | ~15 min each |
| `correct` | top-3 doubtful numbers per page checked by the reader (old) and by the best checker | 2 | a few hours (~3 h per checker) |

The smoke run's timings give a better estimate: training s/it × 750 steps, and the verify bar's s/it × ~200 pages.

## What to read, in order
`results/checker/CHECKER_RESULTS.md`, then:

1. **Table 1, "EDU misread: picks the true value".** Reader ≈ 45% before. The checker must be clearly above 50%; 70%+
   would be a real result. **"EDU correct number: keeps it"** must stay high (≥ 90%), or the checker will break correct numbers.
2. **"FERMAT wrong value: keeps the written one".** The silent-correction resistance. Compare the base model, the reader
   and the checkers.
3. **Table 2: automatic correction.** Success = *Fixed* > *Broken* + *Changed to another wrong value*, and fewer wrong
   numbers per exam after than before. The thresholds are still tuned per student fold for ≤ 1% false changes (`correct.py`).
4. If table 2 is positive, re-run with grading: `LMA_GRADE=1 STAGES=correct sbatch job.sh` (the
   already-verified numbers are reused; only grading runs).

**What to commit:** `git add results/checker` picks up only the safe files: `CHECKER_RESULTS.md`, `eval/**`
(probabilities, no student text), `pairs/pairs_summary.json` (counts), `correct_*/*_metrics.json` and
`correct_*/transcript_diff/*/metrics.json`. Check `git status` before committing.
Pairs, adapters, `verify*.csv`, `decisions.csv` and transcripts quote student work or are large and are git-ignored.

## Fairness of the comparison
- The doubtful numbers come from the same `scores.csv` for both checkers, so both see the same 3 numbers per page.
- The reader baseline is re-run on the same unseen students, so the old 845-number result is not mixed in.
- The scores are the ones from the last correction run (`results/correction_600k/qwen_ftreader/`, 600k pixels); if that
  folder is missing, `results/detect/qwen3b_ft_ftreader/` is used, or the scores are computed (`LMA_DETECT_DIR` overrides).

## Scale-up run (`STAGES=scale`, no training)
The first run checked only the 3 most doubtful numbers per page (106 of 304 misreads), and the options contained the
true value for only 34% of those: the old candidates change one digit at one position, so they can never offer 5.9
for 59, 12 for 123 or 540 for 54. The `scale` stage reuses the trained checker and fixes both limits:

- **More numbers:** the 10 most doubtful per page (`LMA_K`). The checker keeps 99.5% of correct numbers, so checking
  more should add fixes at little cost.
- **Top-k whole numbers as options** (`src/detect/beam_candidates.py`): the reader (verbatim LoRA, transcription prompt)
  reads the page with the transcript fed in up to the number, then searches the number digit by digit (beam width 6; the page is read once, then only digits, "." or "," can follow);
  the 4 best distinct numbers become the first candidates, then the old digit candidates (`LMA_CANDS=4` in total).
  Appearance only: it sees the page and the transcript up to the number, never the rubric or the lines after it.
- The checker judges each candidate against the written value (A/B, both orders), and the same rule decides
  (≤ 1% false changes, tuned on other students). `sweep_correct.py` then tries other rules (error budget, guards).

```bash
LMA_SCALE_LIMIT=5 STAGES=scale sbatch job.sh    # quick test (~20 min): correct_checker_fe_k10_beams_try
rm -rf results/checker/correct_*_try
STAGES=scale sbatch job.sh                      # ~5-6 h on 2 GPUs
cat results/checker/CHECKER_RESULTS.md
```

Read the new row `checker_fe_k10_beams`: *True value among options* (was 34%) and *…among beams* show whether the
beams raised the ceiling; then *Fixed* vs *Broken* + *Changed to another wrong value*. `sweep_metrics.csv` lists other
rules with their `net_fixes`. Commit `results/checker` as before.

## Arbiter, clean FERMAT check, Auditor (`STAGES="arbiter clean_eval audit"`)

**Learned Arbiter** (`src/detect/arbiter.py`, CPU). The scale run showed the options are now good (true value offered
for 69% of the checked misreads) but the decision is weak: on the checker's probability alone, the 1% false-change
budget forces a 0.98 threshold. The Arbiter combines every tool's evidence for each checked number: checker probability
and margin, order agreement, blank-image probability, beam rank and probability, reader doubt (`neg_gap`, `nll`,
`dent`), edit shape and answer-key flags. It's a logistic model trained on other students, whose threshold is the one
with the most net fixes on those students (no change at all unless that is positive; a 1%-budget rule is reported too).
It writes `<run>/arbiter/` with metrics, decisions, corrected transcripts and their transcript diff.

**FERMAT's red markers (a bug found on 12 Nov).** FERMAT's written answers (`pert_a`) mark every planted error as
`\textcolor{red}{..}`. Two consequences:
- The reader's FERMAT metrics compared transcripts with the marked text, so preserved errors counted as silent
  corrections. Re-scored with the markers stripped (`src/finetune/rescore_fermat.py`, `metrics_clean.json`):
  base reader 82.5% → **22.2%** of answers silently corrected, fine-tuned reader 76.3% → **14.0%**; token error
  35.8% → **34.9%** and 16.5% → **8.5%**. Fine-tuning reduces silent correction by more than a third; the old
  conclusion "training barely reduces silent correction" was wrong. `fermat_data.reversion` / `cer` now strip markers.
- The checker's FERMAT pairs kept the marker in the shown line (46% of the silent test pairs had it right at `[?]`),
  a shortcut, and the markers broke the alignment so only 78 silent test pairs were built (382 without). `clean_eval`
  re-measures every model on clean pairs. The EDU numbers are unaffected (no markers there).

**Auditor on FERMAT** (`src/detect/fermat_audit.py`). Our transcripts of FERMAT's test images, numbers labelled ok /
silent correction / misread against what is written. The fine-tuned reader silently corrects only 18 numbers (31 for
the base reader) out of ~8,900, and 50 of 848 planted-error tokens overall (most are letters and operators such as
`a`→`x`, `-`→`+`). So there is little for an Auditor to fix here; `audit` measures the detectors (confidence with the
page and with a blank image, a learned detector held out by problem) and `LMA_AUDIT_FULL=1` also runs beams → checker
→ Arbiter on the flagged numbers. On EDU, the Auditor's tools act as guards: without the blank-image guard no
decision rule can change anything within budget (`sweep_metrics.csv`).

## Next, once the checker works
- The Auditor's own tools: the blank-image test (already computed per change as `p_blank`), the step recompute, and the
  answer-key lookup.
- The agents call the checker through the token ledger (claims name a number by ID and change one token), and the
  fixed rule above decides.
