# Correction Inception: Detecting and Verifying Transcription Errors that Change Grades in AI Grading of Handwritten Exams

**Team Dropouts:** Prabhav (2023115008), Jayant (2023115016), Saloni (2023115001), Sampath (2023102033). LMA Group Project 2026.

**Code:** https://github.com/TOMATOsJr/Correction-Inception. The [`README`](../README.md) on `main` explains how to reproduce every run and which branch holds which part.

---

## Abstract

AI grading of handwritten work usually runs in two steps: a vision-language model (VLM) transcribes the page, and a language model grades the transcript. Reading mistakes then turn into grading mistakes in two directions. A **misread** costs the student marks they earned. A **silent correction** ("correction inception") happens when the reader quietly fixes the student's mistake, giving marks the student did not earn.

We rebuilt the evaluation of EDU-CIRCUIT-HW, a dataset of university circuit-analysis exams, and added a grade-impact metric: the **transcription-caused grading error (TCE)**. TCE is the share of rubric codes whose deduction changes when the same grader reads the AI transcript instead of the expert one. It is 8.1% for Gemini 2.5 Pro and 28.6% for our reader.

**The fine-tuning worked.** Our QLoRA-tuned Qwen2.5-VL-3B reader:
- cuts token errors against what the student wrote from 34.9% to 8.5% on FERMAT;
- lowers the share of answers it silently corrects from 22.2% to 14.0%.

A fine-tuned A/B checker keeps 99.5% of correct numbers (the base model keeps 90.2%).

On top of these models we built:
- detection tools: teacher-forced confidence and a learned selector, which catches 81% of misreads while flagging 44% of numbers;
- correction tools: forced choice, beam candidates, a blank-image guard, an answer-key guard and page consistency;
- adversarial **Advocate/Auditor** agents, in two versions: free-text claims checked by a re-read, and tool-calling agents over a ledger of flagged numbers.

**Correcting automatically, without a human, does not yet work.**
- The best correction on unseen students is net **+2** of 304 misreads.
- Tool-calling agents do *worse* than the same tools without agents (−7 vs +1).
- The Auditor's objections halve the Advocate's damage (−15 → −7).
- In the 80-exam ablation, verified agents leave TCE unchanged (28.2%), and unverified claims make it worse.

We report these negative results with their causes. The main one: the checker picks the true value for only 57% of real misreads, and among the flagged numbers correct ones far outnumber misreads.

## 1. Introduction

Recent work (EDU-CIRCUIT-HW, ACL Findings 2026) shows that even frontier VLMs misread a large share of handwritten university answers: their sample error rates range from 38% to 86%. When a grader uses these transcripts, a wrong digit can cost a student marks for work they did correctly. Silent corrections are harder to spot, because the transcript then looks *better* than what the student wrote.

**Our goal is fairer grades, not a better OCR.** The whole system must be automatic, so a teaching assistant (TA) only sees what the system cannot resolve. We ask four questions:
1. How much grading error is caused by transcription alone (TCE), and in which direction?
2. Can a small reader be fine-tuned to transcribe *verbatim*, i.e. not silently correct?
3. Can the errors that matter be detected, using only the reader's own uncertainty?
4. Can two opposing agents fix them under strict verification, and beat the same tools used without agents?
   - The Advocate argues for the student (misreads); the Auditor argues against unearned marks (silent corrections).
   - Verification uses what is written (a context-free re-read, a forced choice, the student's own work), never mathematical correctness.

**Contributions.**
- A model-agnostic re-implementation of the EDU-CIRCUIT-HW evaluation. It recomputes the paper's reported range of sample error rates (37.6–85.6% vs 37.62–85.43%).
- TCE, a grade-impact metric with student-clustered bootstrap CIs. The paper's version is unreleased.
- A verbatim reader (QLoRA on FERMAT) and an exact silent-correction metric. We found and fixed a red-marker leak that had inflated the first FERMAT numbers (§7).
- Misread detection by teacher-forced token confidence, which works on any transcript, including Gemini's, plus a learned selector.
- Correction tools, each measured on its own against expert transcripts: forced-choice A/B verification, a fine-tuned checker, digit beam candidates, a blank-image guard, an answer-key guard, page consistency and a learned Arbiter.
- Two generations of Advocate/Auditor agents, a full ablation study on the ADA cluster, and a comparison against the same tools without agents.

## 2. Literature review

- **Handwritten STEM recognition and grading.** EDU-CIRCUIT-HW (Georgia Tech ECE 2040) provides expert-rectified transcripts, TA rubric grades and a judge-based error metric (SER/AEC) with a four-way error taxonomy. We reuse its prompts and released transcripts from six VLMs: Gemini 2.5 Pro, Gemini 3 Pro, GPT-5.1, Claude Sonnet 4.5, Qwen3-VL-Plus and Qwen3-VL-8B. The paper discusses but does not release a grade-impact measurement.
- **Over-correction by VLMs.** FERMAT (AI4Bharat) contains handwritten maths answers with planted errors (`pert_a` as written, `orig_a` correct). Results on it show that VLMs often "fix" student errors when transcribing, which is exactly our silent-correction failure.
- **Parameter-efficient fine-tuning.** We use LoRA and QLoRA (a 4-bit base model with low-rank adapters) on Qwen2.5-VL-3B-Instruct, for both the reader and the A/B checker.
- **Self-correction and verification.** Asking an LLM to revise its own output (our baseline B2) is known to be unreliable without external feedback (Huang et al.). Verifier-gated and debate-style multi-agent setups route claims through checks or opposing agents (Du et al.; Irving et al.). We take the debate idea, but a rule-based or learned Arbiter decides from tool evidence, not an LLM.
- **Uncertainty for error detection.** Token-level confidence (log-probability, entropy) is a standard signal for OCR and generation errors. We compute it by teacher forcing, so it also works for closed-model transcripts with no released logits.
- **Tool use by LLM agents.** Our second agent generation calls tools (checker, beams, page counts, blank image, answer key) in a structured format. The aim is for agents to ground their claims in evidence rather than in their own reading of the page.
- **LLM-as-judge.** Judge reliability varies a lot. Our free Qwen-7B judge agreed with Gemini almost not at all (κ = 0.03), so we replaced it with an exact maths-token comparison.

## 3. Datasets used and created

| Dataset | Size | Ground truth | Our use |
|---|---|---|---|
| **EDU-CIRCUIT-HW** | 1,334 answers. Observation set: 513 answers from 11 students with TA marks per rubric code; 293 of them have expert-rectified transcripts. 62 rubrics. | Expert transcripts, TA grades, rubrics, released transcripts from 6 VLMs and the paper's judge verdicts | Main evaluation (TCE, maths diff, misread labels) |
| **FERMAT** | 2,244 handwritten maths answers. Our problem-level split: 1,567 train / 425 test (360 usable for silent correction). | `pert_a` (written, with a planted error), `orig_a` (correct) | Fine-tuning the verbatim reader and the checker; exact silent-correction labels |

**Data we created.**
- **Per-number misread labels.** Each transcript number is aligned to the expert transcript:
  - our Qwen transcripts: 7,701 numbers, 680 misreads (8.8%);
  - Gemini 2.5's transcripts: 10,310 numbers, 116 misreads (1.1%).
- **Per-number FERMAT labels.** For our reader's FERMAT test outputs, each number is labelled ok / misread / silent: 8,916 numbers, 57 misreads, 18 silent corrections.
- **A/B checker training pairs**, of three kinds:
  - silent (FERMAT);
  - look-alike digit swaps (3↔8, 1↔7, …);
  - EDU near-miss misreads from 3 students (student_2, student_35, student_42), which are excluded from every evaluation.
  - Pairs are balanced per source and kind.
- **An 80-exam ablation subset** (`results/ablation/subset_items.csv` on `ablation-results`).

**Caveats.**
- The expert transcripts are corrected *Gemini 2.5 Pro* output, which favours Gemini 2.5.
- The released VLM transcripts came with the dataset; we never called those APIs.
- Student data (transcripts, per-number label and score files, agent ledgers) is never committed; it is in `.gitignore`.

## 4. Methodology

### 4.1 Evaluation pipeline (`src/eval/`)
1. `run_ocr.py` transcribes pages with any backend (Gemini, OpenAI-compatible, or local Qwen + LoRA).
2. `evaluate_recognition.py` reproduces the paper's steps 9–12: judge → "too strict" filter → SER/AEC → taxonomy.
3. `transcript_diff.py` is an exact maths-only comparison against the expert transcript. It covers numbers, operators and variables inside LaTeX maths, and reports wrong numbers per exam.
4. `grade_impact.py` computes **TCE**:
   - the same grader (Qwen2.5-7B-Instruct, 4-bit) marks the expert and the AI transcript against the rubric;
   - every (answer, rubric code) pair whose deduction differs counts as an error;
   - errors are split into *wrongful deduction* and *over-credit*, with student-clustered bootstrap CIs.
   - The grader agrees with TAs 71% of the time; its noise floor is 6.7%.

### 4.2 Verbatim reader (`src/finetune/`)
We fine-tuned Qwen2.5-VL-3B-Instruct with QLoRA on FERMAT, training it to reproduce `pert_a` (what is written) rather than `orig_a`. FERMAT marks the planted error in `pert_a` with `\textcolor{red}{…}`. We strip these markers from the training target and from all metrics (`fermat_data.strip_red`).

The **silent-correction rate** asks: at the spots where `pert_a` ≠ `orig_a`, does the output match the correct version better than the written one?

### 4.3 Misread detection (`src/detect/`)
- **Confidence by teacher forcing.** The transcript under test is fed in as the reader's answer. One forward pass gives each number's `nll`, digit entropy `dent`, `neg_gap` and the reader's preferred alternative digits.
- **Learned selector.** A logistic regression over 28 clues: the number's own doubt, page-level doubt, its rank on the page, its neighbours' doubt, digit shape and look-alike digits. It uses 5-fold cross-validation **split by student**, so every reported number comes from unseen students.
- **Blank-image signal (for the Auditor).** The same transcript is scored again with a white image. A number the reader predicts just as well without the page comes from context, not from the ink. That is the signature of a silent correction.

### 4.4 Correction tools
- **Candidates from appearance only.** Never the answer key, and never "the value that makes the maths right". Three sources:
  - the reader's alternative digits;
  - look-alike swaps (3↔8, 1↔7, 4↔9, …);
  - a **digit-by-digit beam search**: the page is read once, then the beam search runs on the cached keys and values, so it fits an 11 GB GPU.
- **Forced-choice check.** The reader is asked "Is the number marked [?] written as A) x or B) y?", in both option orders to cancel position bias. The A/B probabilities come from the answer-letter logits.
- **Fine-tuned A/B checker.** The reader's LoRA is trained further on the pairs of §3, with the loss on the answer letter only. Two variants: `checker_f` (FERMAT only) and `checker_fe` (FERMAT + 3 EDU students).
- **Guards.** A change is blocked if:
  - the candidate equals the rubric's expected value (answer-key guard), or
  - the checker prefers the candidate almost as much on a **blank image** (blank-image guard).
- **Decision rule.** Change only if both option orders agree and the change probability ≥ τ. τ is set on the other students so that at most 1% of their correct numbers change.
- **Alternatives to that rule:**
  - a **learned Arbiter**: one logistic model over all evidence (checker probability and margin, order agreement, blank-image probability, beam rank and probability, reader doubt, edit shape, answer-key flags), trained on other students;
  - **page consistency**: does the candidate, or the written value, appear elsewhere on the student's page?

### 4.5 Agents (`src/agents/`)

**Generation 1: free-text claims** (branches `agent_interaction`, `ablation`).
- **Roles:** the Advocate looks for misreads that cost marks; the Auditor looks for silent corrections that gave unearned marks. Both are Qwen2.5-7B-Instruct.
- **Claim ledger:** a claim must quote a span and change one number.
- **Tools:**
  - `crop_reread`: a context-free re-read of the line by the fine-tuned reader, with the disputed value masked;
  - consistency with the student's own later lines.
- **Protocols:**
  - *one-shot*: file claims → tools → decide;
  - *interactive*: r1 → tools → r2 → tools → r3, where agents may counter, withdraw, revise or concede.
- **Arbiter:** rule-based with threshold θ = 0.85. It accepts a claim only with supporting evidence, rejects claims the evidence contradicts, and escalates the rest to a TA.

**Generation 2: tool-calling agents over the flagged-number ledger** (branch `ab_checker`, `src/agents/ledger_agents.py`). Generation 1 failed mainly because its claims could not be verified (§7), so generation 2 fixes the claim format:
- **Claims:** agents see a numbered ledger of the page's flagged numbers. A claim names a ledger entry and one candidate value, so every claim can be checked.
- **Tools:** `checker` (A/B probabilities), `beams`, `page_count`, `blank` and `answer_key`, with up to 8 calls per round and 2 rounds per agent.
- **Process:**
  1. the Advocate files claims;
  2. the Auditor objects to any claim it finds unsupported;
  3. a fixed rule applies the surviving claims.
- **Baseline:** the pipeline without agents, using the same tools, the same rule and the same unseen students.

### 4.6 Baselines
- **B1:** no correction.
- **B2:** one-pass LLM self-correction ("fix the transcript").
- **No-verify:** all agent claims applied unchecked.
- **Pipeline:** the same tools and decision rule without agents. The agents must beat this.

## 5. Experiment runs

| Run | Where | Branch | What |
|---|---|---|---|
| Paper reproduction | CPU, from released verdicts | `main` | SER/AEC for 6 VLMs |
| Kaggle runs 1–2 | Kaggle T4 | `main` | TCE for released transcripts; judge calibration |
| PC run | RTX 5070 Ti | `main` | Our reader (untuned and QLoRA); FERMAT eval |
| Agent full run | GPU | `agent_interaction` | 214 exams, interactive agents, 5 variants |
| Colab runs (30 Oct) | Colab | `colab-results` | 50-exam agent run; misread detection |
| Misread detection | ADA | `misread_detection` | Confidence AUROC per transcriber and reader |
| Correction (600k px) | ADA | `forced_choice` | Selector + forced choice + guards, held-out students |
| A/B checker | ADA, 2× 2080 Ti | `ab_checker` | Checker fine-tuning, eval, correction (3, 10 and 20 numbers per page), learned Arbiter, consistency, Auditor on FERMAT and EDU, tool-calling agents |
| Ablation | ADA job 80460, 2× 2080 Ti, 3h23m | `ablation` → `ablation-results` | 80 exams × 2 transcribers; ablations #2, #3, #6, #7, #9, #10, #16 |

### 5.1 Measurement

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

- **Recognition rank doesn't predict grade damage.** Gemini 3 has the best SER but **twice the grade damage** of Gemini 2.5. Part of this is that the answer key also derives from Gemini 2.5.
- **Reading errors go with grading errors.** With the paper's own grader against TA grades, grading disagreement is 59.1% on answers with recognition errors vs 36.8% on answers without.

### 5.2 Fine-tuning

**Verbatim reader on FERMAT** (test split, red markers stripped):

| Reader | Answers silently corrected | Planted-error spots silently corrected | Token error vs written |
|---|---|---|---|
| Qwen2.5-VL-3B base | 22.2% | 8.8% | 34.9% |
| + our QLoRA | **14.0%** | **4.9%** | **8.5%** |

**A/B checker** (held-out pairs; chance = 50%; accuracy averaged over both option orders):

| Pairs | Base | Reader LoRA | checker_f | checker_fe |
|---|---|---|---|---|
| EDU misread: picks the true value (n=196) | 53.1% | 51.5% | 46.9% | **57.1%** |
| EDU correct number: keeps it (n=400) | 90.2% | 95.0% | 99.5% | **99.5%** |
| FERMAT wrong value: keeps the written one (n=382) | 82.2% | 86.9% | **97.9%** | 94.5% |
| FERMAT correct number: keeps it (n=400) | 94.0% | 96.0% | 100% | 99.0% |

- The checker learned to keep what is written, so it resists silent corrections.
- It learned only a little about fixing real misreads: `checker_fe` reaches 57.1% on EDU misreads, and picks it in both option orders 49% of the time (base model: 32%).

### 5.3 Detection
- **Confidence AUROC:** 0.75 (Qwen, ADA), 0.72 (Qwen, Colab), 0.83 (Gemini). The reader's own best alternative equals the true number in only 14–28% of misreads.
- **Learned selector vs confidence cut-off** (held-out students, target recall 80%):

| Transcripts | Detector | Recall | Numbers flagged | Flags that are real |
|---|---|---|---|---|
| Qwen | Learned selector | 81% | **44%** | **29%** |
| Qwen | Confidence cut-off | 80% | 64% | 19% |
| Gemini | Learned selector | 83% | 41% | 2.3% |
| Gemini | Confidence cut-off | 81% | 32% | 2.9% |

About 7% of misreads get ≥ 99.99% confidence, so no threshold catches them at a reasonable cost.

- **Silent corrections on FERMAT (Auditor's detector, our reader):**
  - The blank-image signal separates the 18 silent corrections from the 8,841 correct numbers with AUROC **0.916**. AUROC is reported regardless of the signal's direction; here a lower value means a silent correction.
  - Digit entropy reaches 0.904.
  - Taking the 4 numbers per answer with the most suspicious blank-image score finds 16 of the 18.

### 5.4 Correction of EDU misreads (unseen students)
The 3 checker-training students are left out; 304 misreads remain. *Net* = fixed − changed to another wrong value − correct numbers broken.

| Run | Numbers checked | True value among the options | Fixed | Changed to another wrong value | Correct broken | Net | Misreads |
|---|---|---|---|---|---|---|---|
| Untuned reader, forced choice (`forced_choice`, held-out students) | 845 | – | 2 | 3 | 7 | −8 | – |
| Reader LoRA, 3 per page | 569 | 34% | 2 | 1 | 4 | −3 | 304 → 306 |
| `checker_fe`, 3 per page | 569 | 34% | 5 | 0 | 3 | **+2** | 304 → 302 |
| `checker_fe`, 10 per page + beams | 1,826 | 69% | 10 | 2 | 7 | +1 | 304 → 301 |
| `checker_fe`, 20 per page + beams | 3,224 | 66% | 13 | 2 | 11 | 0 | 304 → 302 |

- With a perfect TA on the `forced_choice` queue (2.2 numbers per page), misreads would fall from 680 to 517.
- Beams double the share of misreads whose true value is offered (34% → 69%). But checking more numbers per page breaks as many correct numbers as it fixes misreads.
- Wrong numbers per exam stay at about 12.96 in every run (12.958 → 12.951 at best).

## 6. Ablations

**80-exam ablation run** (ADA job 80460; transcribers `qwen3b_ft` (ours) and `gemini25`; generation-1 agents). TCE in %, lower is better.

| System | qwen3b_ft | gemini25 |
|---|---|---|
| No correction (B1) | 28.2 (23.0–33.5) | 8.5 (4.4–13.7) |
| B2 one-pass self-correction | **25.0** (20.8–29.4) | **7.5** (4.8–10.2) |
| One-shot agents + Arbiter | 28.2 | 8.5 |
| Interactive agents + Arbiter | 28.2 | 8.5 |
| Accept every claim (no verification) | 28.5 | 8.0 |
| Claims accepted (interactive) | 0 / 112 | 2 / 123 (both harmful) |

- **#9 Headline (system vs baselines).** The agents leave TCE unchanged. B2 lowers it slightly (CIs overlap), but it adds number errors: mean number mismatches rise from 12.5 to 14.7 (qwen) and from 1.4 to 6.9 (gemini).
- **#2 Arbiter evidence** (re-read / consistency / none / accept-all).
  - All verified modes give the same TCE.
  - Accepting every claim unchecked gives claim precision 8.7% (qwen) and 0% (gemini).
  - In the earlier 214-exam run, unchecked claims: 4 helped, 64 hurt, 394 neutral; TCE 28.5 → 29.2.
- **#3 Threshold θ** (0.5–0.95): no effect; at most 2 claims accepted at any θ.
- **#6/#7 Re-reader** (base vs fine-tuned, disputed value masked vs shown). When a re-read resolves, it mostly sides with the transcribed value. This rests on only 12–32 evaluable claims.
- **#10 Single agent.** Advocate-only and Auditor-only give the same TCE as both agents together; each agent files fewer claims alone.
- **#16 Strong vs weak transcriber.** Neither Gemini 2.5 nor our reader benefits.

**Checker-side ablations** (`ab_checker`, unseen students, 304 misreads):

| Ablation | Variants (net) | Finding |
|---|---|---|
| Checker training data | base 53.1%, reader 51.5%, `checker_f` 46.9%, `checker_fe` **57.1%** on EDU misreads | EDU near-miss pairs are needed to learn misreads; FERMAT alone teaches "keep" |
| Candidate source | top-3 digits: true value offered for 34%; + beams at 10 / 20 per page: 69% / 66% | Beams fix the candidate gap, not the choice |
| Numbers per page | 3: **+2**, 10: +1, 20: 0 | More checks add as many broken numbers as fixes |
| Decision rule | simple threshold +2 / +1 / 0; learned Arbiter, net rule −9 to 0; Arbiter, ≤1% budget −10 to 0 | The learned Arbiter ranks better in 3 of 4 runs (AUROC 0.85–0.89 vs 0.81–0.86 for the checker alone), but its threshold doesn't carry over to new students |
| Page consistency | simple +2; veto +2 / +2 / −13; *required* (main rule, fixed in advance) −1 / −11 / −12 | The pattern holds for 12.4% of fixing and 12.6% of harmful changes (10 per page): no separation, because circuit pages reuse the same values |
| **Agents vs pipeline** (generation 2) | see the table below | Agents lose to the pipeline; the Auditor halves the Advocate's damage |

**Generation-2 tool-calling agents (191 pages of unseen students):**

| Run | Pipeline (no agents) | Agents (Advocate + Auditor) | Agents as a filter only | Advocate alone |
|---|---|---|---|---|
| 10 per page + beams | **+1** (10 / 2 / 7) | −7 (3 / 1 / 9) | 0 (1 / 0 / 1) | −15 (6 / 3 / 18) |
| 20 per page + beams | **0** (13 / 2 / 11) | −4 (1 / 1 / 4) | −2 (0 / 1 / 1) | −10 (3 / 1 / 12) |

Net (fixed / changed to another wrong value / correct broken).

- At 10 per page the Advocate filed 569 claims; only 189 were for the value the checker examined, and only 14 would have fixed a misread.
- The Auditor objected to 368 of them.
- 164 claims (29%) failed the format checks.
- Tool use was heavy and mostly well-formed: 825 checker calls, 490 beam calls, 421 page counts, 195 blank-image calls and 82 answer-key calls, with 8 invalid calls.

**Auditor fixes on FERMAT** (our reader; 18 silent corrections, 57 misreads):
- **Simple rule:** changes nothing.
- **Learned Arbiter:** silent corrections 18 → 14, plus 2 misreads fixed and 4 correct numbers broken, so net +2. Answers silently corrected fall from 13.86% to 13.55%.

**Silent corrections in real exams (EDU, via the answer key):**
- If our reader silently corrected toward the right answer, misread transcript values would land on answer-key values more often than correct numbers do. They don't:

  | Reader | Correctly read numbers that are key values | Misreads whose transcript value is a key value | Misreads whose written value is a key value | Misreads toward the key / away from it |
  |---|---|---|---|---|
  | Ours | 79.1% | 56.0% | 63.1% | 89 / 137 |
  | Gemini 2.5 | 77.8% | 42.7% | 50.4% | 14 / 23 |

- The blank-image signal does not carry over: misreads that move toward the key vs correct numbers score AUROC 0.53, compared with 0.92 on FERMAT. Real silent corrections, if any, look different from FERMAT's planted ones.

**Earlier no-API ablations** (mid-submission, Gemini 2.5):
- The arithmetic-inconsistency flag has 50% precision.
- A naive redundancy signal has 3.6% precision, worse than random (5%).
- Unit normalisation triples the number of checkable equalities (45 → 149).

## 7. Evaluation and analysis

**The fine-tuning is the clear gain.** Both fine-tuned models do what they were trained for:
- The reader's token error against what is written falls from 34.9% to 8.5%, about a quarter of the base model's.
- Its silent corrections fall from 22.2% to 14.0% of answers.
- The checker almost never changes a correct number (99.5% kept, vs 90.2% for the base model).
- On EDU these gains show up as TCE 32.9% → 28.6%.

**A measurement bug we found and fixed.** Our mid-submission reported FERMAT silent-correction rates of 82.5% (base) and 76.3% (ours). Those numbers kept FERMAT's red `\textcolor{red}{…}` error markers in the written text, so preserved errors were counted as silent corrections. The markers also leaked into the first checker pairs as a shortcut: 46% of silent test pairs had a marker right at the [?] position. Every figure in this report is recomputed with the markers stripped.

**Why the generation-1 agents had no effect.** The verifier can only check a claim that changes **exactly one number**, and most agent claims changed several numbers or rewrote a line:
- 83–89 of the 112–123 claims ruled on get no reading;
- 157 of 203 `crop_reread` attempts fail (qwen, interactive);
- in the 214-exam run, 653 of 757 re-reads failed.

With no evidence, the Arbiter escalates, so the transcript stays unchanged. The verifier ablations (#2, #3, #6, #7) therefore cannot separate the variants. The run mostly measures this mismatch between claim granularity and the verifier.

**Generation 2 fixed the claim format but still loses to the pipeline.** Every claim is now checkable, and the agents call tools correctly. But they choose worse than the fixed rule:
- **The Advocate over-claims.** It files claims on numbers and values the checker never examined, and drops most of the changes the pipeline would have made correctly.
- **The Auditor is the useful half.** Its objections cut the Advocate's damage roughly in half (−15 → −7 and −10 → −4). Adversarial communication helps, but only relative to an unchecked Advocate.
- **As a filter on the pipeline's changes,** the agents remove good and bad changes alike (net 0 / −2).

**Why the correction tools themselves are weak.** Detection works: AUROC 0.72–0.83 for misreads, 0.92 for FERMAT silent corrections, and the selector cuts flags by about a third at equal recall. Correction is the bottleneck:
- the true value is among the candidates for only 34–69% of misreads;
- even the fine-tuned checker picks it only slightly above chance (57%);
- correct numbers far outnumber misreads among the flagged numbers (3,224 checked vs 265 misreads at 20 per page), so every false change wipes out a fix.

The system follows its design principle "default to keep": it is safe (very few new errors), but not yet useful without a TA.

**Did the architecture beat no architecture?**
- **The fine-tuned models:** yes, clearly.
- **The correction layer on top:** only within noise. The best is +2 of 304 misreads on EDU and +2 numbers on FERMAT. None of these differences are statistically meaningful.
- **TCE after correction** was not re-measured for the `ab_checker` runs. With a net of 0 to +2 numbers, it cannot change measurably (the grader's noise floor is 6.7%).

## 8. Limitations

- **Bias in the ground truth.** The expert transcripts are corrected Gemini 2.5 output, and the answer key and grader also lean on it.
- **Small and skewed data.**
  - EDU has 11 students and 293 expert transcripts; Gemini has only 116 misreads to learn from.
  - The 3 checker-training students are the messiest, so the held-out students are cleaner.
  - FERMAT has only 18 silent corrections for our reader.
- **The grader is an LLM.** Qwen-7B has a 6.7% noise floor and 71% agreement with TAs, so TCE differences of a few points sit inside the CIs. The free LLM judge failed (κ = 0.03) and was replaced by an exact maths diff, which misses non-numeric errors.
- **Scope.** Correction covers numbers only. Variables, operators, dropped lines, diagrams and looping output are not covered.
- **Two Qwen transcript sets.** The ADA and Colab sets differ (8.8% vs 15.5% misreads). The ADA set (`qwen3b_ft`) is the official one: all detection, correction and checker results use it.
- **Ablation run issues.** The "interactive, as built" row is scored on 55/67 exams, not 80.
- **Compute.** Kaggle/Colab T4 and ADA 2080 Ti (11 GB) force a 4-bit grader, a 3B reader and capped image resolution (602,112 pixels).
- **No human labels.** The planned hand-check of about 100 flags was not done.

## 9. Conclusions

- **Grade damage.** Transcription alone changes **8–33% of rubric-level grades**, mostly against the student. The recognition ranking (SER) does not predict grade damage (TCE).
- **Fine-tuning works.**
  - The verbatim reader cuts token errors to about a quarter (34.9% → 8.5%) and silent corrections from 22% to 14% of answers.
  - The A/B checker keeps 99.5% of correct numbers.
- **Detection works reasonably well:** misreads at AUROC 0.75–0.83, FERMAT silent corrections at 0.92 from the blank-image signal.
- **Reliable automatic correction does not yet work:**
  - the best verified correction is within noise (+2);
  - the learned Arbiter and page consistency do not help;
  - tool-calling agents lose to the same tools without agents, although the Auditor halves the Advocate's damage;
  - unverified corrections make TCE worse;
  - only one-pass self-correction lowers TCE, and it adds new number errors.

**Next steps.**
1. Raise the checker's accuracy on real misreads, which is the bottleneck. Options: more EDU near-miss training data, higher-resolution crops, a second reader.
2. Train the Advocate (or its tool-choice policy) on the pipeline's outcomes, instead of prompting it.
3. Hand-label about 100 flags to measure label noise.
4. Evaluate the system as a **TA-triage tool**. That is where it currently adds value: a queue of about 2.2 numbers per page that would fix about a quarter of misreads with a TA.

## 10. References

1. EDU-CIRCUIT-HW: evaluating multimodal models on handwritten university circuit-analysis solutions. arXiv:2602.00095, Findings of ACL 2026. Code: https://github.com/gt-learning-innovation/CIRCUIT_EDU_HW_ACL (Apache-2.0).
2. AI4Bharat. FERMAT: handwritten mathematical answers with planted errors for evaluating VLMs. https://huggingface.co/datasets/ai4bharat/FERMAT (CC BY 4.0).
3. Qwen Team. Qwen2.5-VL Technical Report, 2025.
4. Qwen Team. Qwen2.5 Technical Report, 2024.
5. Dettmers et al. QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 2023.
6. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
7. Huang et al. Large Language Models Cannot Self-Correct Reasoning Yet. ICLR 2024.
8. Du et al. Improving Factuality and Reasoning in Language Models through Multiagent Debate. ICML 2024.
9. Zheng et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023.
10. Irving, Christiano, Amodei. AI Safety via Debate. arXiv:1805.00899, 2018.
11. Schick et al. Toolformer: Language Models Can Teach Themselves to Use Tools. NeurIPS 2023.

---

## Repository and reproducibility

**GitHub:** https://github.com/TOMATOsJr/Correction-Inception

The [`README`](../README.md) on `main` lists every branch, what it adds and how to reproduce each run. Result summaries for the two final runs are copied to `main`:
- `results/checker/CHECKER_RESULTS.md` (A/B checker, correction, Arbiter, consistency, Auditor, agents);
- `results/ablation/ABLATION_RESULTS.md` (the 80-exam ablation).
