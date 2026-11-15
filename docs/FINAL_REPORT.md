# Correction Inception: Verifying Transcription Errors that Change Grades in AI Grading of Handwritten Exams

**Team Dropouts:** Prabhav (2023115008), Jayant (2023115016), Saloni (2023115001), Sampath (2023102033). LMA Group Project 2026.
**Code:** https://github.com/TOMATOsJr/Correction-Inception (see *Repository and reproducibility* at the end for which branch holds what).

---

## Abstract

AI grading of handwritten work usually runs in two steps: a vision-language model (VLM) transcribes the page, and a language model grades the transcript. Reading mistakes then turn into grading mistakes in two directions. A **misread** costs the student marks they earned. A **silent correction** ("correction inception") happens when the reader quietly fixes the student's mistake, giving marks the student did not earn.

We rebuilt the evaluation of EDU-CIRCUIT-HW, a dataset of university circuit-analysis exams, and added a grade-impact metric: the **transcription-caused grading error (TCE)**, the share of rubric codes whose deduction changes when the same grader reads the AI transcript instead of the expert one. TCE is 8.1% for Gemini 2.5 Pro and 28.6% for our QLoRA-tuned Qwen2.5-VL-3B reader. Fine-tuning on FERMAT halves the reader's token errors (16.5% → 8.5% with the markers stripped) and lowers its silent-correction rate from 22.2% to 14.0%.

We then built tools for detecting and correcting errors, and adversarial **Advocate/Auditor** agents whose claims must be verified before a rule-based **Arbiter** applies them:
- A learned misread selector catches 81% of misreads while flagging 44% of numbers (confidence alone needs 64%).
- A fine-tuned A/B forced-choice checker keeps 99.5% of correct numbers, but picks the true value for only 57% of real misreads.

Across seven ablations, **no verified correction method reduced TCE**. Unverified agent claims made grades slightly worse (28.5% → 29.2%). The verified agents changed almost nothing, mostly because their claims did not match what the verifier could check. Only one-pass self-correction lowered TCE (28.2 → 25.0), and it adds new number errors. We report these negative results with their causes.

## 1. Introduction

Recent work (EDU-CIRCUIT-HW, ACL Findings 2026) shows that even frontier VLMs misread a large share of handwritten university answers: their sample error rates range from 38% to 86%. When a grader uses these transcripts, a wrong digit can cost a student marks for work they did correctly. Silent corrections are harder to spot, because the transcript then looks *better* than what the student wrote.

**Our goal is fairer grades, not a better OCR.** We ask four questions:
1. How much grading error is caused by transcription alone (TCE), and in which direction?
2. Can a reader be fine-tuned to transcribe *verbatim*, i.e. not silently correct?
3. Can the errors that matter be detected, using only the reader's own uncertainty?
4. Can two opposing agents fix them under strict verification? The Advocate argues for the student (misreads); the Auditor argues against unearned marks (silent corrections). Verification uses a context-free re-read and the student's own later work, never mathematical correctness. Anything that cannot be verified goes to a human (TA) instead of being changed.

**Contributions.**
- A model-agnostic re-implementation of the EDU-CIRCUIT-HW evaluation. It recomputes the paper's reported range of sample error rates (37.6–85.6% vs 37.62–85.43%).
- TCE, a grade-impact metric with student-clustered bootstrap CIs. The paper's version is unreleased.
- A verbatim reader (QLoRA on FERMAT) and an exact silent-correction metric.
- Misread detection by teacher-forced token confidence, which works on any transcript, including Gemini's. We add a learned selector on top.
- Correction tools: forced-choice A/B verification, a fine-tuned checker, beam candidates, a blank-image guard and an answer-key guard. Each is measured on its own against expert transcripts.
- Advocate/Auditor agents with a rule-based and a learned Arbiter, and a full ablation study on the ADA cluster.

## 2. Literature review

- **Handwritten STEM recognition and grading.** EDU-CIRCUIT-HW (Georgia Tech ECE 2040) provides expert-rectified transcripts, TA rubric grades and a judge-based error metric (SER/AEC) with a four-way error taxonomy. We reuse its prompts and released transcripts from six VLMs: Gemini 2.5 Pro, Gemini 3 Pro, GPT-5.1, Claude Sonnet 4.5, Qwen3-VL-Plus and Qwen3-VL-8B. The paper discusses but does not release a grade-impact measurement.
- **Over-correction by VLMs.** FERMAT (AI4Bharat) contains handwritten maths answers with planted errors (`pert_a` as written, `orig_a` correct). Results on it show that VLMs often "fix" student errors when transcribing, which is exactly our silent-correction failure.
- **Parameter-efficient fine-tuning.** We use QLoRA (4-bit base model with low-rank adapters) on Qwen2.5-VL-3B-Instruct.
- **Self-correction and verification.** Asking an LLM to revise its own output (our baseline B2) is known to be unreliable without external feedback. Verifier-gated and debate-style multi-agent setups route claims through checks or opposing agents. Our design takes the debate idea but makes a rule-based Arbiter, not an LLM, decide from tool evidence.
- **Uncertainty for error detection.** Token-level confidence (log-probability, entropy) is a standard signal for OCR and generation errors. We compute it by teacher forcing, so it also works for closed-model transcripts with no released logits.
- **LLM-as-judge.** Judge reliability varies a lot. Our free Qwen-7B judge agreed with Gemini almost not at all (κ = 0.03), so we replaced it with an exact maths-token comparison.

## 3. Datasets

| Dataset | Size | Ground truth | Our use |
|---|---|---|---|
| **EDU-CIRCUIT-HW** | 1,334 answers. Observation set: 513 answers from 11 students with TA marks per rubric code; 293 of them have expert-rectified transcripts. 62 rubrics. | Expert transcripts, TA grades, rubrics, released transcripts from 6 VLMs and the paper's judge verdicts | Main evaluation (TCE, maths diff, misread labels) |
| **FERMAT** | 2,244 handwritten maths answers; our problem-level split gives 1,567 train / 425 test (360 usable for silent correction) | `pert_a` (written, with a planted error), `orig_a` (correct) | Fine-tuning the verbatim reader and the checker; exact silent-correction labels |

**Data we created.**
- **Per-number misread labels** for 7,701 numbers in our Qwen transcripts (680 misreads, 8.8%) and 10,310 in Gemini 2.5's (116 misreads, 1.1%). Each transcript number is aligned to the expert transcript.
- **A/B checker training pairs.** Three kinds: silent (FERMAT), look-alike digit swaps, and EDU near-miss misreads from 3 students. These 3 students are excluded from every evaluation.
- **An 80-exam ablation subset** (`results/ablation/subset_items.csv`).

**Caveats.**
- The expert transcripts are corrected *Gemini 2.5 Pro* output, which favours Gemini 2.5.
- The released transcripts came with the dataset; we never called those APIs.
- Student data (transcripts, label CSVs) must not be published (see §9).

## 4. Methodology

### 4.1 Evaluation pipeline (`src/eval/`)
1. `run_ocr.py` transcribes pages with any backend (Gemini, OpenAI-compatible, or local Qwen + LoRA).
2. `evaluate_recognition.py` reproduces the paper's steps 9–12: judge → "too strict" filter → SER/AEC → taxonomy.
3. `transcript_diff.py` is an exact maths-only comparison against the expert transcript. It covers numbers, operators and variables inside LaTeX maths.
4. `grade_impact.py` computes **TCE**. The same grader (Qwen2.5-7B-Instruct, 4-bit) marks the expert and the AI transcript against the rubric. Every (answer, rubric code) pair whose deduction differs counts as an error. Errors are split into *wrongful deduction* and *over-credit*, with student-clustered bootstrap CIs. The grader agrees with TAs 71% of the time; its noise floor is 6.7%.

### 4.2 Verbatim reader (`src/finetune/`)
We fine-tuned Qwen2.5-VL-3B-Instruct with QLoRA on FERMAT, training it to reproduce `pert_a` (what is written) rather than `orig_a`. The **silent-correction rate** asks: at the spots where `pert_a` ≠ `orig_a`, does the output match the correct version better than the written one?

### 4.3 Misread detection (`src/detect/`, branches `misread_detection`, `forced_choice`)
- **Confidence by teacher forcing.** The transcript under test is fed in as the reader's answer, and one forward pass gives each number's `nll`, digit entropy `dent`, `neg_gap` and the reader's preferred alternative digits.
- **Learned selector.** A logistic regression over 28 clues: the number's own doubt, page-level doubt, its rank on the page, its neighbours' doubt, digit shape and look-alike digits. It uses 5-fold cross-validation **split by student**, so every reported number comes from unseen students. The cut-off is chosen for 80% recall.

### 4.4 Correction tools (branches `forced_choice`, `ab_checker`)
- **Candidates from appearance only:** the reader's alternative digits, look-alike swaps (3↔8, 1↔7, 4↔9, …) and a digit-by-digit beam search. Never the answer key, and never "the value that makes the maths right".
- **Forced-choice check.** The reader is asked "Is the number marked [?] written as A) x or B) c?" in both option orders, to cancel position bias.
- **Fine-tuned A/B checker.** The reader's LoRA is trained further, with the loss on the answer letter only. There are two variants: `checker_f` (FERMAT only) and `checker_fe` (FERMAT + 3 EDU students).
- **Guards.** A change goes to a TA instead if:
  - the candidate equals the rubric's expected value (answer-key guard), or
  - the reader prefers the candidate almost as much on a **blank image** (blank-image guard), which means it is guessing from context.
- **Decision rule:** change / keep / TA queue. The threshold is set on training students so that at most about 1% of correct numbers change. A **learned Arbiter** combines all tool evidence in one logistic model.

### 4.5 Agents (`src/agents/`, branches `agent_interaction`, `ablation`, `ab_checker`)
- **Advocate:** looks for misreads that cost marks.
- **Auditor:** looks for silent corrections that gave unearned marks.
- Both are Qwen2.5-7B-Instruct and work over a **claim ledger**. A claim must quote a span and change exactly one number.
- **Tools:**
  - `crop_reread`: a context-free re-read of the line by the fine-tuned reader, with the disputed value masked;
  - consistency with the student's own later lines;
  - in `ab_checker`, also the A/B checker, beams, blank-image test and page counts.
- **Protocols:** *one-shot* (file claims → tools → decide) and *interactive* (r1 → tools → r2 → tools → r3; agents may counter, withdraw, revise or concede).
- **Arbiter:** rule-based with threshold θ = 0.85. It accepts a claim only with supporting evidence, rejects claims the evidence contradicts, and escalates the rest to a TA.

### 4.6 Baselines
- **B1:** no correction.
- **B2:** one-pass LLM self-correction ("fix the transcript").
- **No-verify:** all agent claims applied unchecked.
- **Pipeline:** the same tools and rule without agents. The agents must beat this.

## 5. Experiment runs

| Run | Where | Branch | What |
|---|---|---|---|
| Paper reproduction | CPU, from released verdicts | `main` | SER/AEC for 6 VLMs |
| Kaggle runs 1–2 | Kaggle T4 | `main` | TCE for released transcripts; judge calibration |
| PC run | RTX 5070 Ti | `main` | Our reader (untuned, QLoRA); FERMAT eval |
| Agent full run | GPU | `agent_interaction` | 214 exams, interactive agents, 5 variants |
| Colab runs (30 Oct) | Colab | `colab-results` | 50-exam agent run; misread detection |
| Misread detection | ADA | `misread_detection` | Confidence AUROC per transcriber/reader |
| Correction (600k px) | ADA | `forced_choice` | Selector + forced choice + guards, held-out students |
| A/B checker | ADA, 2× 2080 Ti | `ab_checker` | Checker fine-tuning, eval, correction, learned Arbiter, tool-calling agents |
| **Ablation** | ADA job 80460, 2× 2080 Ti, 3h23m | `ablation` → `ablation-results` | 80 exams × 2 transcribers, ablations #2, #3, #6, #7, #9, #10, #16 |

### 5.1 Main results

**Recognition (paper method, 513 samples):**

| Transcriber | SER | 95% CI | AEC |
|---|---|---|---|
| Gemini 3 Pro | 37.6% | 31.7–42.9 | 0.62 |
| Gemini 2.5 Pro | 53.4% | 48.0–59.9 | 1.23 |
| Qwen3-VL-Plus | 61.8% | 53.5–69.7 | 1.38 |
| GPT-5.1 | 71.5% | 62.8–79.5 | 2.05 |
| Claude Sonnet 4.5 | 80.7% | 74.4–87.1 | 2.77 |
| Qwen3-VL-8B (thinking) | 85.6% | 79.7–91.5 | 2.81 |

**Grade impact (TCE, Qwen-7B grader):**

| Transcriber | TCE | Wrongful deduction | Over-credit | Exams affected | Exams with any wrong number |
|---|---|---|---|---|---|
| Gemini 2.5 Pro | 8.1% | 5.5 | 2.6 | 25% | 45% |
| Gemini 3 Pro | 16.4% | 9.9 | 6.5 | 45% | 74% |
| GPT-5.1 | 18.0% | 12.0 | 6.0 | 51% | 83% |
| Qwen2.5-VL-3B, untuned | 32.9% | 27.8 | 5.1 | 73% | 95.6% |
| Qwen2.5-VL-3B, QLoRA (ours) | **28.6%** | 22.8 | 5.8 | 71% | 91.8% |

Gemini 3 has the best SER but **twice the grade damage** of Gemini 2.5. SER and TCE measure different things (the answer key also derives from Gemini 2.5).

**Reading errors and grading errors** (the paper's own grader vs TA grades): grading disagreement is 59.1% on answers with recognition errors vs 36.8% on answers without.

**Verbatim reader on FERMAT** (red error markers stripped):

| Reader | Silent correction | Token error vs written |
|---|---|---|
| Qwen2.5-VL-3B base | 22.2% | 34.9% |
| + our QLoRA | **14.0%** | **8.5%** |

### 5.2 Detection
- **Confidence AUROC:** 0.75 (Qwen, ADA), 0.72 (Qwen, Colab), 0.83 (Gemini). The reader's own best alternative equals the true number in only 14–28% of misreads.
- **Learned selector vs confidence cut-off** (held-out students, target recall 80%):

| Transcripts | Detector | Recall | Numbers flagged | Flags that are real |
|---|---|---|---|---|
| Qwen | Learned selector | 81% | **44%** | **29%** |
| Qwen | Confidence cut-off | 80% | 64% | 19% |
| Gemini | Learned selector | 83% | 41% | 2.3% |
| Gemini | Confidence cut-off | 81% | 32% | 2.9% |

About 7% of misreads get ≥ 99.99% confidence, so no threshold catches them at a reasonable cost.

### 5.3 Correction
**Forced choice with the untuned reader** (held-out students): 845 numbers verified, 207 of them misreads.
- 2 fixed, 3 changed to another wrong value, 7 correct numbers broken.
- 624 sent to a TA (2.2 per page).
- With a perfect TA, misreads fall from 680 to 517.

**A/B checker** (unseen students; chance = 50%):

| Pairs | Base | Reader | checker_f | checker_fe |
|---|---|---|---|---|
| EDU misread: picks the true value | 53.1% | 51.5% | 46.9% | **57.1%** |
| EDU correct number: keeps it | 90.2% | 95.0% | 99.5% | 99.5% |
| FERMAT wrong value: keeps written (markers removed) | 82.2% | 86.9% | **97.9%** | 94.5% |
| FERMAT correct number: keeps it (markers removed) | 94.0% | 96.0% | 100% | 99.0% |

The checker learned to keep what is written, so it stops silent corrections. It barely learned to fix real misreads.

**Automatic correction:**
- Best configuration (`checker_fe`, 10 numbers per page with beam candidates): 10 fixed, 2 changed to another wrong value, 7 broken. Misreads 304 → 301.
- The learned Arbiter (AUROC 0.87) gives a net change of −9 to +0.

## 6. Ablations

Most ablations come from the full ADA run (job 80460): 80 exams, transcribers `qwen3b_ft` (ours) and `gemini25`. The checker-side ablations come from `ab_checker`. TCE in %, lower is better.

| System | qwen3b_ft | gemini25 |
|---|---|---|
| No correction (B1) | 28.2 (23.0–33.5) | 8.5 (4.4–13.7) |
| B2 one-pass self-correction | **25.0** (20.8–29.4) | **7.5** (4.8–10.2) |
| One-shot agents + Arbiter | 28.2 | 8.5 |
| Interactive agents + Arbiter | 28.2 | 8.5 |
| Claims accepted (interactive) | 0 / 112 | 2 / 123 (both harmful) |

- **#9 Headline (system vs baselines).** The agents leave TCE unchanged. B2 lowers it slightly (CIs overlap), but it adds number errors: for Gemini, exams with a number error rise from 48.8% to 71.2%.
- **#2 Arbiter evidence** (re-read / consistency / none / accept-all).
  - All verified modes give the same TCE.
  - Accepting every claim unchecked: 28.5% (qwen) and 8.0% (gemini), with claim precision 8.7% and 0%.
  - In the earlier 214-exam run, unchecked claims: 4 helped, 64 hurt, 394 neutral; TCE 28.5 → 29.2.
- **#3 Threshold θ** (0.5–0.95): no effect; at most 2 claims accepted at any θ.
- **#6/#7 Re-reader** (base vs fine-tuned, masked vs not). When a re-read resolves, it mostly sides with the transcribed value. On Gemini, the default `ft_mask` has the lowest verifier accuracy (72.7%) and `base_nomask` the highest (100%). This rests on only 12–32 evaluable claims.
- **#10 Single agent.** Advocate-only and Auditor-only give the same TCE as both agents together; each agent files fewer claims alone.
- **#16 Strong vs weak transcriber.** Neither Gemini 2.5 nor our reader benefits.
- **Checker ablations** (`ab_checker`):
  - FERMAT-only vs FERMAT + EDU training: see §5.3.
  - Candidate sources: top-3 vs 10 per page with beams. Beams lift the share of fixable numbers (true value among the options) from 34% to 69%, but also break more correct numbers.
  - Consistency rules (simple / required / veto / consistency-only): best net +2 (*veto*); the pre-registered *required* rule gives −1 to −11.
  - Tool-calling agents vs the pipeline without agents: on the small smoke set both are net 0 / −1.
- **Earlier no-API ablations** (mid-submission, Gemini 2.5):
  - The arithmetic-inconsistency flag has 50% precision.
  - A naive redundancy signal has 3.6% precision, which is worse than random (5%).
  - Unit normalisation triples the number of checkable equalities (45 → 149).

## 7. Evaluation and analysis

**Why the agent system has no effect.** The verifier can only check a claim that changes **exactly one number**, and most agent claims change several numbers or rewrite a line:
- 83–89 of the 112–123 claims ruled on get no reading;
- about 155 of ~200 `crop_reread` calls fail with "needs a claim that changes exactly one number";
- in the 214-exam run, 653 of 757 re-reads failed.

With no evidence, the Arbiter escalates, so the transcript stays unchanged. The ablations of the verifier (#2, #3, #6, #7) therefore cannot separate. The run mostly measures this mismatch between claim granularity and the verifier.

**Why the tools themselves are weak.** Detection works (AUROC 0.72–0.83; the selector cuts flags by about a third at equal recall). Correction is the bottleneck:
- the true value is among the candidates for only 34–69% of misreads;
- even the fine-tuned checker picks it barely above chance (57%);
- correct numbers outnumber misreads about 3:1 among the flagged numbers, so every false change wipes out a fix.

This is consistent with the design principle "default to keep": the system is safe (almost no new errors), but it is not yet useful without a TA.

**Where the value is today.**
- Measurement: TCE plus exact silent-correction labels.
- The verbatim reader: half the token errors, and fewer silent corrections.
- Triage: the selector + guards produce a TA queue of about 2.2 numbers per page. With a perfect TA, about a quarter of misreads are fixed (680 → 517).

## 8. Limitations

- **Bias in the ground truth.** The expert transcripts are corrected Gemini 2.5 output, and the answer key and grader also lean on it. Gemini-sourced candidates are switched off for this reason.
- **Small and skewed data.** EDU has 11 students and 293 expert transcripts. Gemini has only 116 misreads to learn from. The 3 checker-training students are the messiest, so the held-out students are cleaner.
- **The grader is an LLM.** Qwen-7B has a 6.7% noise floor and 71% agreement with TAs, and TCE differences of a few points sit inside the CIs. The free LLM judge failed (κ = 0.03) and was replaced by an exact maths diff, which misses non-numeric errors.
- **Scope.** Correction covers numbers only. Variables, operators, dropped lines, diagrams and looping output are not covered.
- **Two Qwen transcript sets.** The ADA and Colab sets differ (8.8% vs 15.5% misreads); one has to be fixed as official.
- **Ablation run issues.**
  - The "interactive, as built" row is scored on 55/67 exams, not 80.
  - Some error lines in the log are not yet inspected.
  - The agent claim format and the verifier do not match, so most verifier ablations are uninformative.
- **Compute.** Kaggle/Colab T4 and ADA 2080 Ti (11 GB) force a 4-bit grader, a 3B reader and capped image resolution.
- **No human labels yet.** About 100 claims were planned for hand-checking label noise and the judge.

## 9. Conclusions

- Transcription alone changes **8–33% of rubric-level grades**, mostly against the student.
- The recognition ranking (SER) does not predict grade damage (TCE).
- Fine-tuning a small reader to be verbatim works: token errors fall by about half and silent corrections from 22% to 14%.
- Detecting *where* errors are works reasonably well.
- Reliably *fixing* them without a human does not yet work:
  - verified corrections, whether from the agents, the fixed pipeline or the learned Arbiter, leave TCE unchanged;
  - unverified corrections make it worse;
  - only one-pass self-correction lowers TCE, by adding other errors.

**Next steps.**
1. Make agents file single-number claims tied to selector IDs, or split multi-number claims, then re-run from the `agents` stage (everything is resumable).
2. Improve candidate recall (higher-resolution re-reads, a second reader).
3. Hand-label about 100 flags.
4. Evaluate the system as a **TA-triage tool**, which is where it currently adds value.

## 10. References

1. EDU-CIRCUIT-HW: evaluating multimodal models on handwritten university circuit-analysis solutions. arXiv:2602.00095, Findings of ACL 2026. Code: https://github.com/gt-learning-innovation/CIRCUIT_EDU_HW_ACL (Apache-2.0).
2. FERMAT: AI4Bharat. Handwritten mathematical answers with planted errors for evaluating VLMs. https://huggingface.co/datasets/ai4bharat/FERMAT (CC BY 4.0).
3. Qwen Team. Qwen2.5-VL Technical Report, 2025.
4. Qwen Team. Qwen2.5 Technical Report, 2024.
5. Dettmers et al. QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 2023.
6. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
7. Huang et al. Large Language Models Cannot Self-Correct Reasoning Yet. ICLR 2024.
8. Du et al. Improving Factuality and Reasoning in Language Models through Multiagent Debate. ICML 2024.
9. Zheng et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023.
10. Irving, Christiano, Amodei. AI Safety via Debate. arXiv:1805.00899, 2018.

*(Before submitting, check the exact titles and authors of refs 1–2 against the papers.)*

---

## Repository and reproducibility

**GitHub:** https://github.com/TOMATOsJr/Correction-Inception

| Branch | Last update | Keep for the final? | Contents |
|---|---|---|---|
| `main` | 25 Oct | Base | Eval pipeline, reader fine-tuning, baselines, Kaggle/PC results, mid-submission |
| `agent_interaction` | 29 Oct | Superseded | Interactive Advocate/Auditor, 214-exam agent run (contained in all branches below) |
| `misread_detection` | 30 Oct | Superseded | Confidence scoring, labels, detection metrics (merged into `forced_choice`) |
| `colab-results` | 30 Oct | **Results only; contains student data** | 50-exam Colab agent run, detection runs, Qwen transcripts |
| `forced_choice` | 8 Nov | Superseded | Selector, forced-choice verification, correction pipeline (contained in `ab_checker`) |
| `ab_checker` | 13 Nov | **Yes: most complete code** | All of the above + fine-tuned A/B checker, learned Arbiter, tool-calling agents, `CHECKER_RESULTS.md` |
| `ablation` | 30 Oct | **Yes: ablation code** | `docs/ABLATION.md`, `scripts/run_ablation.sh` (branches off `agent_interaction`) |
| `ablation-results` | 15 Nov (on ADA, not pushed yet) | **Yes: ablation results** | Job 80460 results, `scripts/ablation_full.slurm`, `docs/ABLATION_RUN_2026-11-15.md` |

**Suggested final layout:** merge `ab_checker` and `ablation-results` into `main` (they share the `agent_interaction` base, so expect conflicts in `src/agents/`). Point the report at `main`.

**README must add** (assignment requirement):
- environment setup: Python 3.11 venv, torch cu124, `requirements-gpu.txt`;
- data download and layout (`LMA_DATA`, EDU-CIRCUIT-HW, FERMAT with `HF_TOKEN`);
- one command per experiment: `run_ocr` / `grade_impact`, `scripts/run_detect.sh`, `job.sh` (checker), `sbatch scripts/ablation_full.slurm`;
- expected outputs and run times;
- a results table that links to the `metrics.json` files.

**Data handling:** the `colab-results` and `agent_interaction` branches contain student transcripts and label CSVs. Remove them before the repository is graded.
