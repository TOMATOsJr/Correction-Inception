# Ablation run, 15 Nov 2026 (ADA)

Full results: `results/ablation/ABLATION_RESULTS.md`. Per-variant TCE: `results/grade_impact/abl_*/metrics.json`.

## Run

| | |
|---|---|
| Job | Slurm 80460, `scripts/ablation_full.slurm`, gnode078 (2 × RTX 2080 Ti, 11 GB) |
| Time | 02:39 – 06:02 IST, 3h23m, exit 0 |
| Exams | 80 (`results/ablation/subset_items.csv`), spread over 11 students |
| Transcribers | `qwen3b_ft` (ours), `gemini25` (Gemini 2.5 Pro) |
| Models | Agents, Arbiter and grader: Qwen2.5-7B-Instruct (4-bit). Reader: Qwen2.5-VL-3B-Instruct + `results/reader_lora` |
| Smoke test | 5 exams on gnode061, passed (`results/ablation_smoke/`, not committed) |

## Headline (TCE %, lower is better)

| System | qwen3b_ft | gemini25 |
|---|---|---|
| No correction | 28.2 (23.0–33.5) | 8.5 (4.4–13.7) |
| B2 one-pass self-correction | **25.0** (20.8–29.4) | **7.5** (4.8–10.2) |
| One-shot agents + Arbiter | 28.2 | 8.5 |
| Interactive agents + Arbiter | 28.2 | 8.5 |
| Claims accepted (interactive) | 0 / 112 | 2 / 123 (both harmful) |

## Findings per ablation

- **#9 Headline.** The agent system leaves TCE unchanged for both transcribers. Only B2 self-correction lowers TCE (−3.2 and −1.0 points, overlapping CIs), and it adds number mismatches (Gemini: exams with a number error 48.8% → 71.2%).
- **#2 Evidence.** All Arbiter evidence modes give the same TCE. Accepting every claim unchecked barely moves TCE (28.5%, 8.0%), and those claims are mostly wrong (claim precision 8.7% and 0%).
- **#3 Threshold θ.** No effect from 0.5 to 0.95: at most 2 claims accepted at any θ.
- **#6/#7 Reader.** When a re-read resolves, it mostly sides with the transcribed value. Masking the disputed value does not help: on Gemini, `ft_mask` (the system default) has the lowest verifier accuracy (72.7%), `base_nomask` the highest (100%). Only 12–32 evaluable claims, so weak evidence.
- **#10 Single agent.** Solo Advocate / Auditor give the same TCE as the joint run; each files fewer claims alone.
- **#16 Strong vs weak transcriber.** Neither benefits.

## Why the agent system has no effect

The verifier can only check a claim that changes **exactly one number**. Most agent claims change several numbers or rewrite a line:

- 83–89 of the 112–123 claims ruled on get **no reading** (note: "unverifiable: not a single-number change, or line/page not found").
- 155–157 of ~200 agent `crop_reread` calls fail, mostly with "crop_reread needs a claim that changes exactly one number"; a few with "span not found in transcript".

With no reading, the Arbiter escalates, so every variant leaves the transcript unchanged and #2/#3/#6/#7 cannot separate. The run therefore mainly measures this mismatch between claim granularity and the verifier.

**Suggested fix before a re-run:** make agents file single-number claims (prompt / protocol change in `src/agents/`), or split multi-number claims into single-number sub-claims before verification. A re-run only needs the `agents` stage onward; everything is resumable.

## Open points

- "Interactive, as built" is scored on 55 (qwen) and 67 (gemini) exams, not 80. Check why before quoting it.
- The run log has 13 lines matching "error" although the job exited cleanly; not yet inspected.

## Reproducing on ADA

```bash
sbatch scripts/ablation_full.slurm
```

Setup that the script relies on (one-time):

- Python 3.11 venv at `~/lma-venv311` (the old py3.9 venv cannot install `transformers==5.20.2`): torch from the `cu124` index, then `requirements-gpu.txt`.
- Data at `~/Correction-Inception/data/edu_text/edu_data`, including `Observationset_Final/v6_Gemini_2p5` (from `lma_bundle.zip`).
- `/share1` is not mounted on compute nodes and has a ~3000-file quota; the job downloads models into node-local `/scratch/$USER/hf_cache` (compute nodes have internet).
- Partition `u22` has a **1-hour default time**: always pass `--time`. QoS `medium` allows up to 4 days, 4 GPUs. `gnode069` had a GPU fault and is excluded.
