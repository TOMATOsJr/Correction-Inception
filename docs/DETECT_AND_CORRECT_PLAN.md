# Misread detection and correction: results so far and plan

Branch `forced_choice`. Scope: **misreads** (the transcriber wrote a different number than the student did).
Silent corrections (the transcriber confidently writes the *right* value over the student's *wrong* one) need
different detectors and are listed at the end.

The rule throughout: **every tool is measured alone against the expert transcripts before agents may use it.**
The crop re-read skipped this step and failed in the full agent run (0 accepted corrections).

---

## 1. Where we are

### 1.1 Confidence scoring (built: `src/detect/score_tokens.py`)

The local reader (Qwen2.5-VL-3B, optionally with our LoRA adapter) sees the page image and the transcription prompt.
Instead of generating, the transcript under test is fed in as the answer (teacher forcing), and one forward pass
gives the reader's probability for every token. Works for any transcript, including Gemini's (no logits released).

Per number: `nll` (doubt in the written value), `neg_gap` (how far a different token is preferred), `ent`,
`dent` (uncertainty among the 10 digits), `alt` / `alts` (the reader's preferred replacement digits).

### 1.2 Labels (built: `src/detect/labels.py`)

Each transcript number is aligned with the expert transcript: `misread` = a number aligned to a different number.

| Transcriber | Numbers | Misreads |
|---|---|---|
| Qwen 3B fine-tuned (Ada transcripts) | 7,701 | 680 (8.8%) |
| Qwen 3B fine-tuned (Colab transcripts) | 6,884 | 1,068 (15.5%) |
| Gemini 2.5 Pro | 10,310 | 116 (1.1%) |

The two Qwen transcript sets differ; one must be agreed as the official one.

### 1.3 Detection quality (`src/detect/evaluate_detection.py`)

AUROC (0.5 = random): Qwen 0.75 (Ada) / 0.72 (Colab), Gemini 0.83. Beats random and the number-length baseline in
every full run. The reader's own best alternative equals the true number in only 14–28% of misreads.

Recall needs to be high (target **70–80% of misreads**). With confidence alone (Qwen, Colab scores):

| Flag if P(written number) < | Numbers flagged | Misreads caught | Flags that are real |
|---|---|---|---|
| 0.99 | 38% | 64% | 26% |
| 0.999 | 64% | 80% | 19% |
| 0.9999 | 88% | 93% | 16% |

About 7% of misreads get ≥ 99.99% confidence: no confidence threshold reaches them at a sensible cost.

---

## 2. Better selector: a learned misread probability (planned)

**Problem.** One confidence cut-off treats every page alike. 0.95 is suspicious on a clean page and normal on a
messy one, so reaching 80% recall flags 65% of all numbers.

**Method.**

1. **Clues per number** (CPU, from the existing scores and transcript; no new GPU work):
   - own doubt: `nll`, `neg_gap`, `ent`, `dent`, `n_pieces`
   - page context: page mean of `nll` / `dent`, the number's rank within its page, page length, position on page
   - neighbours: doubt of the previous and next maths token
   - shape: number of digits, decimal point, which digits occur (look-alikes 3/8, 1/7, 4/9, 5/6, 0/6)
   - the reader's alternative differs by exactly one digit
2. **Train** a logistic regression (boosted trees did no better) to predict `misread` from the clues, using the
   expert labels.
3. **Evaluate on unseen students only**: 5-fold split by student (train on 4/5 of students, test on the rest,
   rotate). Never report numbers from students the model was trained on.
4. **Score** every number with the learned misread probability.
5. **Pick the cut-off** for the target recall (80%) on training students, and apply it to the held-out ones.

**First result** (Colab Qwen scores, held-out students):

| Detector | Recall | Numbers flagged | Flags that are real |
|---|---|---|---|
| Confidence only | 80% | 65% | 19% |
| Learned (28 clues) | 80% | **41%** | **30%** |
| Confidence only | 90% | 85% | 16% |
| Learned (28 clues) | 90% | **60%** | **23%** |

Strongest clues: page length, digit uncertainty, page-level doubt, position, neighbours' doubt. Misreads cluster on
pages the reader struggles with overall.

**To do.** Code it as `src/detect/selector.py`. Re-run it on the Ada scores and on Gemini transcripts, whose
errors look different. Later, add more clues:
- a second reading: the other transcriber, the base vs the fine-tuned reader, sampled transcripts;
- re-scoring flagged lines at higher resolution;
- a hand check of about 100 flags for label noise, which caps precision.

The selector's job is **recall** (miss nothing). Precision is the corrector's job (section 3).

---

## 3. Correcting the flagged numbers (planned; step 3.2 built)

Principle: **default to "keep as written".** A correct number that the corrector changes is a new error, and in
the flagged set correct numbers outnumber misreads about 3:1. The unchecked agent run shows the danger: of 462
claims, 4 helped, 64 hurt and 394 changed nothing, and grading errors rose from 28.5% to 29.2%.

For each flagged number `x`:

### 3.1 Candidates, from appearance only
- the reader's top other digits at the weakest digit (`alts`)
- single look-alike digit swaps (3↔8/5, 1↔7, 4↔9, 5↔6/3, 0↔6/8, 2↔7)
- the other transcriber's value at the aligned position, if it differs
- the reader's re-read of the line at higher resolution (top few readings)

**Never** the rubric / answer key, and never "the value that makes the maths right": that is how silent
corrections are made. If the true value is not among the candidates it cannot be fixed, so measure
`candidate_recall` first.

### 3.2 Forced-choice check (built: `src/detect/verify_choice.py`)
"Is the number marked [?] written as A) x or B) c?" The line is shown with the number masked. The answer is read
from P('A') vs P('B'), and the question is asked again with the options swapped to cancel position bias.
`p_c` = mean over both orders. `consistent` = both orders pick the same winner.

### 3.3 Guards against silent corrections
Reject the change (send it to a TA) if:
- the candidate equals the rubric's expected value at that step (the Auditor-side check; the rubric is never shown
  to the reader);
- the blank-image test shows the reader prefers the candidate almost as much **without** the page, i.e. it is
  guessing from context rather than reading.

### 3.4 Decision rule (thresholds set on training students)

| Evidence | Action |
|---|---|
| consistent and `p_c` ≥ τ_high, no guard fired | **change** x → c |
| `p_c` ≤ τ_low in both orders | **keep** x |
| anything else (inconsistent, middling, guard fired) | **TA queue**, with the evidence attached |

τ_high is chosen so that `false_change_rate` ≤ about 1% on training students. The consistency of the student's
own later lines (e.g. line 5 uses 3) is recorded as supporting evidence for the TA, not used to decide: it rewards
"making sense", which is the silent-correction bias.

### 3.5 Measuring it

| Metric | Target |
|---|---|
| `candidate_recall`: true value is in the list | high; the ceiling for everything below |
| `fix_rate`: misread changed to the true value | as high as possible |
| `wrong_fix_rate`: misread changed to another wrong value | low |
| `false_change_rate`: correct number changed | ≤ ~1% |
| TA load: flags sent to a TA per page | report it |
| Transcript error rate after correction (`transcript_diff`) | below `none` |
| Grading errors (TCE) after correction (`grade_impact`) | below `none` (28.5%) |

The correction only counts as a success if transcript errors **and** grading errors go down compared with no
correction.

### 3.6 Then the agents
Once 3.1–3.4 pass on their own, they become the agents' tools:
- the Advocate gets the misread tools; the Auditor gets the silent-correction guards and the rubric;
- claims refer to selector IDs and change one token each, so whole-line rewrites and quoted spans are impossible;
- the decision rule of 3.4 decides, not the agents' opinion.

The agents must beat the fixed pipeline (3.1–3.4 with no agents) on the same exams, or they are not worth their cost.

---

## 4. Order of work

1. Run `scripts/run_detect.sh` on Ada with a visible GPU (`LMA_LIMIT=10` first). For high recall, set
   `LMA_VERIFY_K=15` until the selector exists. Push `verify_metrics.json`.
2. Build `selector.py` (section 2); verify on the Ada and Gemini scores.
3. Point `verify_choice.py` at the selector's flags; add the other-transcriber candidates and the 3.3 guards;
   set τ_high / τ_low on training students.
4. Apply the corrections and measure transcript and grading error (3.5).
5. Wire the tools into the agents (3.6) and compare against the fixed pipeline.

## 5. Not covered by this plan: silent corrections

Confidence cannot see these (the reader is sure). Planned detectors:
- **blank-image test**: score with the page vs a blank image (also the 3.3 guard);
- **answer-key match**: a transcript value equal to the rubric's value where the student's working implies otherwise;
- **step check**: recompute the student's own arithmetic; a correct result that does not follow from their previous
  line is suspicious.

Also open: variable/operator misreads (same scores, not yet evaluated), dropped lines, looping output.

## 6. Data handling

`labels.csv`, `scores*.csv`, `verify*.csv`, transcripts and agent logs quote student work and must not be pushed.
Student data is currently on the public `colab-results` and `agent_interaction` branches and must be removed.
