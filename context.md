# Context handover: LMA Group Project 2026, "Correction Inception" (Team Dropouts)

Read this first in a new chat. Last updated **2026-10-25**.

## 1. Who / what / when
* **Team "Dropouts":**
  * Prabhav 2023115008
  * Jayant 2023115016
  * Saloni 2023115001
  * Sampath 2023102033 (the user)
* **Project "Correction Inception"** (proposal accepted; PDF and LaTeX in `docs/proposal/`).
  * AI grading of handwritten exams = a VLM **transcribes** the page, then an LLM **grades** the text.
  * Misreads cause **wrongful deductions**.
  * The reader silently "fixing" student mistakes causes **over-credit**. We call this "correction inception".
  * Our system has two adversarial LLM agents:
    * the **Advocate** finds misreads that cost the student marks;
    * the **Auditor** finds silent corrections that gave unearned marks.
  * A **rule-based Arbiter** (not an LLM) accepts, rejects or escalates each claim to a human. It uses two kinds of evidence:
    * a **context-free re-read** by a fine-tuned faithful reader;
    * **consistency with the student's own later work** (never mathematical correctness).
  * The goal is fairer grades, not "a better OCR".
* **Deadlines:**
  * **Mid-submission 2026-10-30**: progress, datasets, baselines, ablations, results, challenges, mitigation, timeline, code.
  * **Final 2026-11-15.**
  * Evaluations Nov 16–19.
* **User rules:**
  * project folder = `C:\Users\hp\OneDrive\Desktop\LMA Project`;
  * **no GitHub until the end** (git initialised, nothing committed);
  * **free tools only**: no paid APIs, no card;
  * never print or commit `.env`.
  * The user wants **layman-level explanations** when asked, honest status, and step-by-step instructions.
* **User's machine:** Windows, i5, 16 GB RAM, **no NVIDIA GPU**, Python 3.12. GPU work runs on Kaggle (T4, free) or a **friend's RTX 5070 Ti PC** (one-visit, limited access; it turned out to run Linux).

## 2. Data and paths
* **EDU-CIRCUIT-HW** (arXiv 2602.00095, Findings of ACL 2026; Georgia Tech ECE 2040).
  * Data at `D:\lma\data\EDU-CIRCUIT-HW_v1` (outside OneDrive); set as `EDU_DATA` in `.env`. Paper code is in `external/CIRCUIT_EDU_HW_ACL`.
  * 1,334 answers. **Observation set 513** (11 students) has TA marks per rubric code (`gt_per_code`) and their AI grader's marks (`pred_per_code`). The test set is 821/828.
  * **293** observation answers have **expert-rectified transcripts**. The experts corrected *Gemini 2.5 Pro's* output, which is a possible bias in its favour. List: `src/eval/items_with_expert_transcript.csv`.
  * Rubrics: `Screenshot_output_anon/rubric_outputs/P<q>.json`, 62 problems, codes like M/E/C/U/NC.
  * **Released transcripts** from Gemini 2.5 Pro, Gemini 3 Pro, GPT-5.1, Claude Sonnet 4.5, Qwen3-VL-Plus and Qwen3-VL-8B are in `Observationset_Final/v6_*`. **We never called those APIs: these came with the dataset.**
  * The paper's judge verdicts (Gemini 2.5 Pro) are also released.
* **FERMAT** (ai4bharat/FERMAT on Hugging Face, gated, CC-BY-4.0): 2,244 handwritten maths answers.
  * Fields: `pert_a` (as written, with a planted mistake), `orig_a` (correct).
  * Our problem-level split is sha-hash based: train 1,567 used / test **425** (360 evaluable for silent correction).
* **Upload bundles in `D:\lma\kaggle_upload\`:**
  * `lma_src.zip`: code;
  * `edu_text.zip`: transcripts, rubrics, splits, paper code + verdicts; no images;
  * `edu_circuit_hw_images.zip`: 592 MB, 1,290 pages;
  * `llm_cache.sqlite`.
* **Single PC bundle:** `D:\lma\FOR_DRIVE\lma_bundle.zip` (~600 MB). It extracts to `C:\lma\{START_HERE.md, LMA Project\, edu_text\, images\}`.
* ⚠️ A copy `LMA Project\lma_bundle.zip` sits in the OneDrive folder (student data syncing). Tell the user to delete it again if it's still there.
* `D:\lma\ci_eval` = old duplicate, contains a `.env` copy; offer deletion.
* Test leftovers that are safe to delete: `D:\lma\{ks, localsim, tiny, agent_test, agent_test_real}`.

## 3. Pipeline roles (as explained to the user)
* **Transcriber:** image → Markdown/LaTeX.
  * Ours: Qwen2.5-VL-3B, untrained and QLoRA-trained (`results/pc_run/results/reader_lora`).
* **Grader:** rubric + transcript → deductions per code.
  * Ours: Qwen2.5-7B-Instruct, 4-bit, `hf-text:` backend, batched.
  * **Works:** agrees with TAs 71%; noise floor 6.7%.
* **Judge:** lists reading mistakes vs the expert transcript (the paper's SER/AEC).
  * The paper used Gemini 2.5 Pro.
  * **Our free Qwen-7B judge FAILED:** "no errors" on 509/513 transcripts, κ = 0.03 vs Gemini. Dropped.
  * Replaced by `src/eval/transcript_diff.py`, an exact maths-only comparison against the expert: numbers/operators/variables inside `$..$`, `\[..\]`.

## 4. Metrics
* **Paper:**
  * **SER** = % of answers with ≥ 1 judge-listed mistake (after the "too strict" DELETE/KEEP filter);
  * **AEC** = mean number of mistakes;
  * 4-type taxonomy: symbol/char, structure/notation, diagram, text/logic;
  * judge errors count as 1.
* **Ours:**
  * **TCE (grade impact)** = % of (answer, rubric code) pairs whose deduction differs when the *same* grader marks the expert transcript vs the AI transcript. Split into wrongful deduction (AI version marked lower) and over-credit (marked higher). Student-clustered bootstrap CIs.
  * **Maths diff:** % of answers with any differing number, mean number mismatches, token error rate.
  * **FERMAT silent-correction rate:** at the spots where `pert_a` ≠ `orig_a`, does the output match the correct version better than the written one?
  * **Token error rate vs written.**

## 5. Results so far (all real)
**Paper reproduction** (from the released verdicts; `results/baseline_reference/`):
* SER 37.6% (Gemini 3), **53.4% (Gemini 2.5 Pro, AEC 1.23)**, 61.8% (Qwen3-VL-Plus), 71.5% (GPT-5.1), 80.7% (Claude 4.5), 85.6% (Qwen3-VL-8B).
* The paper reports 37.62–85.43%. **This is the same, not better.**

**Grade impact (TCE), Qwen-7B grader** (Kaggle run 2, `results/kaggle_run2/`; PC run `results/pc_run/`):

| Transcriber | TCE | wrongful | over-credit | exams affected |
|---|---|---|---|---|
| Gemini 2.5 Pro (released) | 8.1% | 5.5 | 2.6 | 25% |
| Gemini 3 Pro (released) | 16.4% | 9.9 | 6.5 | 45% |
| GPT-5.1 (released) | 18.0% | 12.0 | 6.0 | 51% |
| Our Qwen-VL-3B, untrained | 32.9% | 27.8 | 5.1 | 73% |
| Our Qwen-VL-3B, trained | **28.6%** | 22.8 | 5.8 | 71% |

* Finding: Gemini 3 has the best SER but **2× the grade damage** of Gemini 2.5. Caveat: the answer key derives from Gemini 2.5.

**Maths diff, % of exams with any differing number:**
* Gemini 2.5: 45%
* Qwen3-VL-Plus: 70%
* Gemini 3: 74%
* GPT-5.1: 83%
* ours untrained: 95.6% (28.9 wrong numbers per exam)
* ours trained: 91.8% (**13.7**)

**FERMAT (our reader):**
* silent correction **82.5% → 76.3%** (untrained 83.6% on Kaggle);
* token error vs written **35.8% → 16.5%**.

**Story:** training halves reading errors but barely reduces silent correction, and does not reduce over-credit. That motivates the agents.

**Older no-API analyses:**
* `shipped_grading_analysis`: grading disagreement 59.1% with vs 36.8% without recognition errors.
* `arith_consistency`: 5 student slips, all preserved by Gemini 2.5; 7 misreads detected, 50% flag precision.
* `redundancy_detector`: precision 3.6% vs 5% random. A negative ablation.

## 6. Code map (`LMA Project/`)
* **`src/eval/`**
  * `llm.py`: `call_llm`; backends gemini / `openai:` / `hf-text:` / mock.
    * `hf-text` = batched, one copy per GPU, token budget `HF_TOKEN_BUDGET`, OOM → one-by-one retry.
    * sqlite cache; the mock also fakes agent claims.
  * `evaluate_recognition.py`: paper steps 9–12, plus `normalize_judge_output` for non-Gemini judges.
  * `run_ocr.py`: `HFReader` with `read_with_conf` (confidence = geometric-mean token probability); `--items/--shard/--repetition-penalty 1.05/--max-new-tokens 2048`; `LMA_DEADLINE`.
  * `grade_impact.py`: TCE. Its parser salvages the `deductions` JSON. Always writes to `results/grade_impact/<name>`.
  * `transcript_diff.py`, `calibrate_judge.py`, `edu_compat.py`.
* **`src/finetune/`**
  * `fermat_data.py`: `load_fermat()`; env `LMA_FAKE_FERMAT=n` gives synthetic test data.
  * `train_reader.py`: QLoRA; `warmup_steps=0.03` (transformers 5 removed `warmup_ratio`); Windows `dataloader_num_workers=0`; `LMA_NO_QUANT` = plain-LoRA fallback.
  * `eval_fermat.py`.
* **`src/agents/`**
  * `core.py`: `Claim`; `span_check`; `changed_number` (claims may quote context but must change exactly one number); `consistency_check`; `arbitrate` (theta 0.85); `apply_accepted` (dedupes); `test_core.py` has 7 passing tests.
  * `agents.py`: prompts, plus the hints section via `agent_prompt()`.
  * **`run_agents.py` (NEW, the agent runner).** Phases:
    * `claims`: LLM agents, hint = numeric equalities that don't hold;
    * `reread`: the trained reader re-reads the claimed line with the disputed number replaced by `?`, then `resolve()` → proposed/span;
    * `decide`: Arbiter → variants `none / agents / noverify / advocate_only / auditor_only`, written in the transcriber folder layout as `<qid>_markdown.md`. Also produces a theta sweep and **claim precision vs the expert transcript**;
    * `selfcorrect`: baseline B2, one-pass "fix the transcript".
* **`src/analysis/`**: the no-API analyses.
* **`scripts/`**
  * `run_all.py`: one-shot GPU runner. Preflight (bitsandbytes failure = warning only), priority order, resume, `--hours`, QLoRA → plain-LoRA fallback, failure isolation, logs, auto-zip `local_results.zip`. Test overrides: `LMA_READER`, `LMA_GRADER=mock`, `LMA_ALLOW_CPU`.
  * `setup_local.ps1`: venv at `<Data>\v`, the short path matters on Windows.
  * `make_notebooks.py`: Kaggle notebooks 01/02/03/05. Use raw strings: heredoc `\n` escaping broke it several times.
  * `package_for_kaggle.py`.
* **`docs/`**:
  * `mid_submission.md`: **needs updating with section 5's results**;
  * `LOCAL_RUN.md`, `START_HERE.md` (PC instructions);
  * `proposal/`, `SETUP_accounts.md`.
* **Tested library pins** (`requirements-gpu.txt`): transformers 5.20.2, peft 0.21.2, accelerate 1.15.0, datasets 5.1.0, bitsandbytes 0.50.2; torch cu128 for the RTX 50-series.
* **CPU test environment:** `<scratchpad>/venv` from the old session, which may be gone. A tiny random Qwen2.5-VL is at `D:\lma\tiny` (for code tests). Install torchvision too, and keep paths short (WinError 206).

## 7. Where we stopped (in progress)
The user chose **option B: get a first real agent result into the mid-submission.**

**Done:**
* `run_agents.py` written; lint clean.
* Full mock pipeline over all 293 real PC transcripts (`results/pc_run/results/Outputs/qwen3b_ft`): 586 agent calls → 584 re-reads → decide → scoring.
* The "none" variant reproduces the PC maths diff exactly (13.659), so the layout and scoring are identical.
* Fixed bugs: duplicate fixes applied twice; multi-number spans; `.env` not loaded.
* The arithmetic hint finder only fires on 9/293 exams, and the samples looked like false alarms. Weak hints.

**Not done:**
1. **A real-quality check of the agent prompts.** I proposed 12 calls to the user's Gemini free tier (`gemini-3.1-flash-lite`, `--limit 6`). **The user declined that tool call and has not decided A (use Gemini) vs B (test only on the PC).** Ask again.
2. **The PC/GPU orchestrator for agents** (e.g. a `scripts/run_agents_all.py` reusing `run_all.Run`/preflight; skip the FERMAT download). Steps per target:
   * claims (7B, ~586 calls);
   * reread (3B + adapter);
   * decide (CPU);
   * for each variant: `transcript_diff` + `grade_impact` (~10 min each on the 5070 Ti);
   * then `selfcorrect` (B2) last.

   **Targets:**
   * `qwen3b_ft` (transcripts from `pc_run`): ship them, plus `reader_lora` (153 MB), in the new bundle;
   * `gemini-2.5-pro` (`edu_data/Observationset_Final/v6_Gemini_2p5`).

   Estimate ~3–5 h on the 5070 Ti; use `--hours`.
3. A new bundle + START_HERE for the second PC visit (target ≤ Oct 28), and a prompt for Claude on that PC. The previous prompt is in this chat's history: run in a separate window, no code changes, never paste the token.
4. **Write `docs/mid_submission.md`** with all results, the honest framing and the ablations, then team review, then submit Oct 30.
5. **Final (Nov 15):**
   * baselines B2–B6;
   * full ablations (no re-read, no consistency, theta, untuned re-reader);
   * human labels (~100 claims, ~15 h of team effort);
   * failure analysis; final report.

## 8. Lessons / gotchas
* Kaggle: T4, 12-hour sessions, 30 GPU-hours a week. Importing a notebook drops its inputs. vLLM 0.32 is broken on the T4 image. The log viewer doesn't refresh tqdm. **One account per person** (ToS).
* Gemini: 2.5 Pro/Flash return 404 for new users; the free tier is ~20 requests/day per model.
* Windows: PowerShell 5.1 treats stderr as an error under `Stop` (use exit codes). The cp1252 console crashes on Unicode (`reconfigure(encoding="utf-8")`). Long paths break pip.
* The 3B reader loops on long pages without a repetition penalty (penalty for exam reading only; FERMAT kept at 1.0 for a fair before/after).
* Always state **SER ≠ TCE**: different metrics. Never claim we beat the paper.
