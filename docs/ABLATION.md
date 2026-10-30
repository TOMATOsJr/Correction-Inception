# Ablations on ADA (branch `ablation`)

Seven ablations of the interactive agent system, run on a **subset of 80 exams** (configurable) for two transcribers.
One script runs everything: `scripts/run_ablation.sh`, submitted with `scripts/ablation_ada.slurm`.

| # | Question | How it is run | Where the answer is |
|---|---|---|---|
| 2 | Which evidence does the Arbiter need? | Arbiter with re-read + consistency / re-read only / consistency only / nothing (`noverify`) | `ABLATION_RESULTS.md` § #2 |
| 3 | How does θ trade precision against human workload? | Decisions + claim precision at θ = 0.5 … 0.95; θ = 0.5, 0.7, 0.85, 0.95 also scored for TCE | § #3 |
| 6 | Does fine-tuning the reader make verification better? | Every claim re-read by the fine-tuned and the base Qwen2.5-VL-3B | § #6/#7 |
| 7 | Does the reader copy the value it is shown? | Re-read with the disputed value hidden (`?`) vs shown | § #6/#7 |
| 9 | Headline: does the system beat simple baselines? | none vs B2 self-correction vs one-shot agents vs interactive agents | § #9 |
| 10 | Does having an opponent change an agent's claims? | Advocate alone and Auditor alone (separate debates) vs their claims taken out of the joint run | § #10 |
| 16 | Does it help strong or only weak transcribers? | Everything above for `qwen3b_ft` (ours) and `gemini25` (Gemini 2.5 Pro, released) | each section per transcriber, summary in § #16 |

## What changed on this branch

1. **Bug fix: the interactive re-read could never support a claim.** `crop_reread` returned the reader's whole line
   (`V = 3 * 4 = 12`), and the Arbiter compared it with the claimed span (`3 * 4`). They almost never matched, so in the
   full 293-exam run 104 of 129 successful re-reads were unresolvable and **0 claims were accepted**. The one-shot path
   already mapped readings with `resolve()`; the interactive path now does too (`make_reread` in `run_agents.py`).
   Without this fix every ablation would show "no change".
2. **Arbiter evidence switches** (`ArbiterConfig.use_reread`, `use_consistency` in `core.py`) for #2.
3. **`verify` phase:** the Arbiter re-reads *every* claim it rules on with one reader configuration, whether or not the
   agent asked for a re-read. All systems and readers are therefore judged by the same verifier. Configurations:
   `ft_mask` (the system default), `base_mask`, `ft_nomask`, `base_nomask`.
4. **`ablate` phase (CPU):** the Arbiter for every evidence mode × θ × reader config; writes the corrected transcripts
   to `variants/` and `ablate_summary.json`. It also records **verifier accuracy**: how often the re-read sides with the expert transcript.
5. **`--agents advocate`** runs one agent alone (#10), and **`--items CSV`** runs on a subset.
6. `HF_MAX_NEW_TOKENS` (llm.py) so self-correction can output a whole transcript instead of stopping at 1024 tokens.

The original `decide` still runs too, and its output is scored as **`agenttools`**: the system as built, using only
the evidence the agents asked for.

## Before you start (once)

Same setup as the earlier agent run (`scripts/run_agents_all.sh`). Check that these exist, relative to the repo root:

```bash
ls ../venv/bin/activate                                    # the venv used before
ls ~/Correction-Inception/lma_data/edu_text/edu_data/Screenshot_output_anon   # dataset
ls local_results/results/Outputs/qwen3b_ft | head -3       # our fine-tuned reader's transcripts
ls local_results/results/reader_lora                       # the LoRA adapter
ls ~/Correction-Inception/lma_data/edu_text/edu_data/Observationset_Final/v6_Gemini_2p5 | head -3
```

If any of them live elsewhere, set `LMA_VENV`, `LMA_DATA`, `LMA_QWEN_ROOT`, `LMA_ADAPTER` or `LMA_GEMINI_ROOT`. The models
(Qwen2.5-7B-Instruct, Qwen2.5-VL-3B-Instruct) are already in the HF cache from the earlier runs. `HF_HOME` defaults to
`/scratch/$USER/hf_cache`; point it at the existing cache if that is somewhere else (the old scripts used
`/scratch/jayanth.raju/hf_cache`).

## Steps

```bash
# 1. get the branch
cd ~/Correction-Inception
git fetch origin && git checkout ablation && git pull

# 2. unit tests (CPU, seconds)
source ../venv/bin/activate
python src/agents/test_core.py && python src/agents/test_protocol.py

# 3. smoke test on 5 exams, to catch path problems and to time one exam (interactive node with 2 GPUs)
LMA_N=5 LMA_OUT=results/ablation_smoke LMA_GI_PREFIX=smoke bash scripts/run_ablation.sh
cat results/ablation_smoke/ABLATION_RESULTS.md        # every table should have numbers
rm -rf results/ablation_smoke results/grade_impact/smoke_*   # then throw the smoke away

# 4. the real run (80 exams), as a batch job
sbatch scripts/ablation_ada.slurm          # or: LMA_N=100 sbatch scripts/ablation_ada.slurm
squeue -u $USER                            # job state
tail -f ablation_<jobid>.out               # progress

# 5. results
cat results/ablation/ABLATION_RESULTS.md
```

**Resuming.** Every stage skips finished work, so after a crash or the time limit, just submit the same command again.
To run only some stages: `STAGES="verify ablate score collect" sbatch scripts/ablation_ada.slurm`.
The subset is sampled once and saved in `results/ablation/subset_items.csv`. Delete it only if you want different exams.

**Timing.** Use the smoke run to estimate. Almost all the time is spent in the `agents` stage: 3 debates + one-shot +
self-correction per transcriber, so 2 × 5 runs over N exams. `verify` is ~2 short re-reads per exam per configuration.
`score` re-grades only the exams whose transcript actually changed, because grades are cached. Roughly: agents-stage
time on the smoke ÷ 5 × 80 ≈ the real run.

## Stages

| Stage | What it does | GPUs |
|---|---|---|
| `subset` | 80 exams with an expert transcript and both transcribers' outputs, spread evenly over the 11 students | CPU |
| `agents` | per transcriber: `interactive`, `solo_advocate`, `solo_auditor` (7B on GPU 0, reader on GPU 1), then `oneshot` and `selfcorrect` (7B on both) | 2 |
| `verify` | GPU 0: `ft_mask` re-reads for every run. GPU 1: `base_mask`, `ft_nomask`, `base_nomask` for the interactive run | 2, in parallel |
| `ablate` | `decide` (interaction stats) + `ablate` (all Arbiter variants) | CPU |
| `score` | `transcript_diff` + `grade_impact` (Qwen-7B grader) for every variant | 2 |
| `collect` | writes `results/ablation/ABLATION_RESULTS.md` | CPU |

## Output layout

```
results/ablation/
  subset_items.csv                 the exams used
  ABLATION_RESULTS.md              all tables
  <transcriber>/                   qwen3b_ft, gemini25
    agents_llm_cache.sqlite        agent calls (solo runs reuse the joint run's round-1 calls, so round 1 is identical)
    interactive/  solo_advocate/  solo_auditor/    r1/r2/r3/tools.jsonl, decide_summary.json, verify_<tag>.jsonl,
                                                   ablate_summary.json, variants/<variant>/, transcript_diff/
    oneshot/                       claims.jsonl, verify_ft_mask.jsonl, ablate_summary.json, variants/
    selfcorrect/Outputs/selfcorrect/
results/grade_impact/abl_<transcriber>_<run>__<variant>/metrics.json     TCE per variant
```

Variant names: `<evidence>__<reader>[__t<θ>]`, e.g. `full__ft_mask` (the system), `reread_only__ft_mask`,
`full__base_nomask`, `full__ft_mask__t0.5`, plus `none`, `noverify`, `posthoc_advocate__ft_mask`.
`variants/` and `Outputs/` are git-ignored: they copy student transcripts and can be rebuilt from the logs
(`STAGES=ablate`). Commit the `*.json` summaries, `metrics.json` files and `ABLATION_RESULTS.md`.

## Troubleshooting

- **CUDA out of memory in re-reads** (the earlier full run had ~60): `LMA_REREAD_MAX_PIXELS=602112 STAGES="verify ablate score collect" sbatch ...`.
  Failed verify rows are retried automatically on the next run.
- **Only one GPU granted:** the `auto` phase needs 2. Ask for `-G 2` (already in the SLURM file).
- **Offline plumbing test without GPUs** (what was used to check this branch): build a tiny fake dataset and run with
  `LMA_LLM=mock LMA_READER=mock LMA_GRADER=mock LMA_GI_PREFIX=dryrun`. Afterwards delete
  `results/grade_impact/dryrun_*`. The numbers are meaningless; the point is that every stage runs.

## Reading the results honestly

- 80 exams from 11 students gives wide CIs. Report the student-bootstrap CI, and treat differences that fall inside it as "no detectable difference".
- The expert transcripts were made by correcting Gemini 2.5 Pro's output, so `gemini25` is favoured in #16.
- Claim precision and verifier accuracy use the expert transcript **only for evaluation**; the system never sees it.
- If the Arbiter still accepts almost nothing after the fix, the bottleneck is claim quality (claims that don't change
  exactly one number, or quote spans not in the transcript). The agent-tool stats table shows this.
