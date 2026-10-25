# Mid-Submission Report: Correction Inception (Team Dropouts)
Due: 30 October 2026 · Final: 15 November 2026 · Evaluations: 16–19 November 2026

> Required by the TAs: progress, datasets, baseline experiments, ablations, results, challenges, mitigation plans,
> timeline, plus an updated GitHub repo. Sections marked **TODO** still need filling in before submission.

## 1. Progress summary
- [x] Proposal accepted.
- [x] Reproduced the EDU-CIRCUIT-HW evaluation pipeline. It is model-agnostic and reuses the paper's prompts unchanged (`src/eval/`).
- [x] Validated the pipeline: it recomputes the paper's reported range of sample error rates (37.6–85.6% vs 37.62–85.43%).
- [x] Dataset downloaded (9.2 GB) and organised; data kept off OneDrive (`D:/lma/data`), configured via `.env`.
- [x] **Free baseline grading analysis** (no API): reading errors are linked to grading errors (Section 3.3).
- [x] **Arithmetic-consistency study** on real data: silent corrections vs misreads (Section 3.4).
- [x] Grade-impact (TCE) script implemented (the paper's version is unreleased) and smoke-tested.
- [x] Correction Inception core implemented and tested offline: claim ledger, verification tools (span check, consistency with the student's own later work + value-reuse signal, never correctness), rule-based arbiter, Advocate/Auditor agents (`src/agents/`, 7 unit tests + end-to-end demo).
- [x] Kaggle notebooks + packaged data for GPU work (untuned Qwen baseline, QLoRA fine-tuning, FERMAT silent-correction metric).
- [ ] ~~Live reproduction of Gemini 2.5 Pro~~: **blocked**. Google no longer offers gemini-2.5-pro to new API users (HTTP 404). Its transcripts and judge outputs ship with the dataset, so we use those.
- [ ] **TODO** Untuned Qwen2.5-VL-3B transcription and evaluation (reader baseline).
- [ ] **TODO** Faithful-reader QLoRA fine-tuning on FERMAT (first run).
- [ ] **TODO** Grade-impact measurement: the transcription-caused grading error (TCE) rate, i.e. our week-2 power gate.

## 2. Datasets
| Dataset | Size | Ground truth | Our use |
|---|---|---|---|
| EDU-CIRCUIT-HW | 1,334 university circuit-analysis answers (513 with expert verbatim transcripts and grades, 11 students; 821 with grades only, 18 students) | Expert transcripts, expert grades, rubrics | Main evaluation; split by student (3 dev / 8 eval) |
| FERMAT | 2,244 handwritten school maths solutions with planted errors | `pert_a` (as written), `orig_a` (correct) | Fine-tuning the verbatim reader; exact silent-correction labels |

## 3. Baseline experiments
### 3.1 Recognition baseline (paper method: Gemini 2.5 Pro judge, steps 9–12)
| Transcriber | SER | 95% CI (student bootstrap) | AEC |
|---|---|---|---|
| Gemini 3 Pro | 37.6% | 31.7–42.9 | 0.62 |
| **Gemini 2.5 Pro** | **53.4%** | 48.0–59.9 | **1.23** |
| Qwen3-VL-Plus | 61.8% | 53.5–69.7 | 1.38 |
| GPT-5.1 | 71.5% | 62.8–79.5 | 2.05 |
| Claude Sonnet 4.5 | 80.7% | 74.4–87.1 | 2.77 |
| Qwen3-VL-8B (thinking) | 85.6% | 79.7–91.5 | 2.81 |
| Qwen2.5-VL-3B, untuned (ours to run) | **TODO** | | |
| Qwen2.5-VL-3B, fine-tuned verbatim reader | **TODO** | | |

Error taxonomy for Gemini 2.5 Pro: 46.8% symbol/character, 26.9% structure/notation, 16.3% diagram, 10.0% logical flow.

### 3.2 Silent-correction baseline (FERMAT test, exact labels)
**TODO**: error-reversion rate, i.e. how often the transcript matches `orig_a` instead of `pert_a` at the planted error. Compare Gemini, untuned Qwen and fine-tuned Qwen.

### 3.3 Do reading errors cause grading errors? (paper's grader vs TA labels, no API)
The paper's LLM grades (`pred_per_code`, graded from the Gemini 2.5 Pro transcript) are compared with TA-verified grades (`gt_per_code`), split by whether the paper's judge found recognition errors in the transcript.

| Transcript | Samples | Sample-level grading disagreement | 95% CI (student bootstrap) | Code-level disagreement | Over-credit | Wrongful deduction |
|---|---|---|---|---|---|---|
| All | 513 | 48.7% | 40.9–57.8 | 16.3% | 2.0% | 14.3% |
| **With recognition errors** | 274 | **59.1%** | 50.4–67.7 | 19.9% | 2.3% | 17.6% |
| **Without recognition errors** | 239 | **36.8%** | 27.0–48.3 | 12.1% | 1.6% | 10.5% |

Answers whose transcript contains reading errors are graded wrongly much more often (59% vs 37%). This is correlational evidence that transcription errors propagate into grades, and it is the core motivation of the project. Most disagreements are wrongful deductions (the grader is harsher than the TA).

### 3.4 Silent corrections vs misreads in real university data (arithmetic consistency, no API)
293 samples with expert transcripts. We extract purely numeric equalities, check which ones the student got wrong, and align them with the Gemini 2.5 Pro transcript.

| Quantity | Value |
|---|---|
| Checkable numeric equalities (gold) | 149 (~0.5 per sample) |
| Genuine student arithmetic slips (gold) | 5 |
| …preserved faithfully by Gemini 2.5 Pro | **5 / 5** (0 arithmetic silent corrections) |
| Misreads that broke the student's arithmetic | **7** (e.g. dropped minus signs; `9/4` read as `9`) |
| Precision of the "arithmetically inconsistent" flag on real transcription differences | 50% (7/14) |

Findings:
- On this university dataset, *arithmetic* silent corrections by Gemini 2.5 Pro are rare. Over-correction reported on school maths (PINK, on FERMAT) may be less common here, or may take symbolic/conceptual forms. We will measure it directly on FERMAT (exact labels) and with the LLM judge on EDU.
- Arithmetic checks catch real misreads but cover only ~0.5 equalities per answer, because circuit work is mostly symbolic. This supports our design: an agent-built value graph (symbol links, reused values) is needed beyond plain arithmetic.

### 3.5 Grading baselines B1–B6
`src/eval/grade_impact.py` is implemented: it grades the gold and target transcripts with the same grader and rubric. A smoke test on 5 samples gave a grader noise floor of 4% code flips and 84% code-level agreement with the TA. The full run is pending API quota (see Challenges).

## 4. Ablations
### 4.1 Done (no API): which consistency signal localises real misreads?
Evaluated on 293 Gemini 2.5 Pro transcripts against expert transcripts (226 aligned numeric differences).

| Evidence signal | Needs | Precision | Recall | Comment |
|---|---|---|---|---|
| Arithmetic inconsistency of a numeric equality | parser only | **50%** (7/14) | low (≈0.5 checkable equalities per answer) | Catches dropped signs and broken fractions |
| Naive redundancy: orphan number with a repeated confusable variant | parser only | 3.6% | 7.1% | **Worse than random (5.0%)**: sign and decimal variants co-occur legitimately |
| Unit handling in the parser (off → on) | n/a | n/a | 45 → 149 checkable equalities | Unit/prefix normalisation triples coverage |

Conclusion: structure-free signals are too weak. Redundancy only helps once we know *which value is a copy of which*. This motivates the agent-built value graph (symbol linking, value reuse) planned for the final system.

### 4.2 Planned (need GPU/API)
- Reader: untuned vs fine-tuned Qwen2.5-VL-3B (FERMAT silent-correction rate, EDU SER/AEC).
- Context-free vs full-page re-read.
- System: −value-graph consistency, −re-reader, −Advocate, −Auditor, −retrieval, and claims accepted without tool verification.
- Judge: gemini-3.5-flash vs the paper's judge verdicts (calibration).

## 5. Results
**TODO**: fill in from `results/*/metrics.json`.

## 6. Challenges
- The paper's released evaluation code stops after 20 samples (a debug cap); our version evaluates all 513.
- The paper's grade-impact code is not released, so we are implementing it ourselves.
- The reference transcript falls back to Gemini 2.5's own output where experts made no correction, which slightly favours Gemini 2.5.
- The judge is itself an LLM, so it must be validated against ≥100 human labels.
- Compute: no local GPU, so GPU work runs on Kaggle/Colab T4 (QLoRA ≈ 20 h). API rate limits apply.
- **gemini-2.5-pro (the paper's transcriber and judge) is unavailable to new API users** (HTTP 404), and Pro models are quota-blocked on the free tier.
- **Free tier = 20 requests/day per model.** A full judge/grader run needs ~1,500 calls, so it is impossible on the free tier.
- Circuit answers are mostly symbolic, so numeric consistency checks cover few lines.

## 7. Mitigation plans
- Cache every API call. Use the paper's released outputs for the Gemini baseline to save budget.
- Use gemini-2.5-flash for debugging and gemini-2.5-pro only for official runs.
- Run the 4-bit/QLoRA configuration with capped image resolution to fit T4 memory.
- Validate the LLM judge on human labels; report results with and without the judge filter.
- Use the shipped Gemini 2.5 Pro transcripts and judge verdicts, so no re-transcription is needed.
- Use **gemini-3.5-flash** as judge/grader, and **calibrate** it against the paper's shipped judge verdicts on identical transcripts (`src/eval/calibrate_judge.py`: agreement, Cohen's κ, SER difference).
- Enable billing or education credits for the full runs (estimated ≪ $95 with Flash pricing + caching).
- All GPU work is on Kaggle (notebooks in `notebooks/`, data packaged by `scripts/package_for_kaggle.py`).

## 8. Timeline to final submission (15 Nov)
| Dates | Work |
|---|---|
| 23–30 Oct | Live baseline reproduction, untuned Qwen eval, FERMAT splits, start fine-tuning, TCE (B1) → **mid-submission** |
| 31 Oct – 5 Nov | Fine-tuned reader eval; consistency checker; confidence head; baselines B2–B4 |
| 6–11 Nov | Agents + arbiter; B5, B6; full ablations; judge validation on human labels |
| 12–15 Nov | Final runs, failure analysis, report, code clean-up → **final submission** |
| 16–19 Nov | Evaluations |
