"""Grade impact of transcription errors: the transcription-caused grading error (TCE) rate.

The EDU-CIRCUIT-HW paper reports this analysis, but its code is not released, so this is our implementation
(it is also baseline B1 in our proposal and the proposal's week-2 power gate).

For each sample with an expert-verified transcript:
  1. grade the GOLD transcript   (expert-rectified)            -> deductions per rubric code
  2. grade the TARGET transcript (a model's transcription)      -> deductions per rubric code
  3. same grader, same rubric => any difference is CAUSED BY TRANSCRIPTION.

Unit = (sample, rubric code). A code is a TCE if target and gold deductions differ:
  * wrongful deduction : target deducts MORE than gold  (misread costs the student marks)
  * over-credit        : target deducts LESS than gold  (e.g. a silent correction hides an error)
Also reported: sample-level TCE rate, score MAE, grader validity vs TA labels (gt_per_code),
grader noise floor (re-grading the gold transcript), and student-clustered bootstrap CIs.

Example:
  python grade_impact.py --target-root <EDU_DATA>/Observationset_Final/v6_Gemini_2p5 --target-model-dir gemini-2.5-pro \
         --grader gemini-3.5-flash --name B1_gemini25 --repeats 1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_recognition import (DEFAULT_EDU_REPO, KEYS, PROJECT_ROOT, RECTIFIED, load_env,  # noqa: E402
                                  load_markdown)
from llm import Cache, call_llm  # noqa: E402

GRADER_PROMPT = """You are a strict but fair teaching assistant grading a university circuit-analysis homework answer.
Grade ONLY with the rubric below. The problem statement is not available; infer the task from the rubric and the student's work.
The student's handwritten work has been transcribed to text. Grade exactly what the transcript shows.

### RUBRIC (JSON; each criterion has a code and deduction rules)
{rubric}

### STUDENT SOLUTION (transcript)
{transcript}

### INSTRUCTIONS
- For each rubric code, decide the total deduction (0 if nothing applies), never exceeding that code's max_deduction.
- Follow-through: do not deduct again for later steps that correctly use an earlier wrong value.
- Return ONLY JSON: {{"deductions": {{"<code>": <number>, ...}}, "reasons": {{"<code>": "<rule id + short reason>"}}}}
  Include every rubric code in "deductions" (use 0 when no deduction).{nonce}
"""


def rubric_path(data_root: Path, qid: str) -> Path:
    return data_root / "Screenshot_output_anon" / "rubric_outputs" / f"P{qid.replace('_', '.', 1)}.json"


def parse_json(text: str) -> dict | None:
    for cand in (text, *re.findall(r"```(?:json)?(.*?)```", text, re.S), *re.findall(r"\{.*\}", text, re.S)):
        try:
            return json.loads(cand.strip())
        except Exception:
            continue
    return None


def parse_codes(s) -> dict:
    try:
        d = json.loads(s) if isinstance(s, str) and s.strip() else {}
        return {k: float(v) for k, v in d.items()}
    except Exception:
        return {}


def grade(transcript: str, rubric: dict, grader: str, cache: Cache, nonce: str = "") -> dict | None:
    prompt = GRADER_PROMPT.format(rubric=json.dumps(rubric, ensure_ascii=False, indent=1),
                                  transcript=transcript, nonce=f"\n(run {nonce})" if nonce else "")
    out = call_llm(prompt, grader, temperature=0.0, cache=cache)
    js = parse_json(out) if not out.startswith("Error:") else None
    if not js or "deductions" not in js:
        print(f"[grade failed] {out[:160]!r}", file=sys.stderr)
        return None
    codes = [c["code"] for c in rubric.get("criteria", [])]
    caps = {c["code"]: float(c.get("max_deduction", 1e9)) for c in rubric.get("criteria", [])}
    d = {}
    for c in codes:
        try:
            d[c] = min(max(float(js["deductions"].get(c, 0) or 0), 0.0), caps[c])
        except Exception:
            d[c] = 0.0
    return d


def boot_ci(values: pd.Series, groups: pd.Series, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    g = [v.values for _, v in values.groupby(groups)]
    if len(g) < 2:
        return None
    s = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(n)]
    return [round(100 * float(np.percentile(s, 2.5)), 2), round(100 * float(np.percentile(s, 97.5)), 2)]


def main():
    load_env(DEFAULT_EDU_REPO)
    import os
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=os.environ.get("EDU_DATA"))
    p.add_argument("--items", default=None, help="default: obsetf_involved_data.csv (has TA gt_per_code)")
    p.add_argument("--target-root", required=True, help="folder containing Homework_collected_database_trial_* for the target")
    p.add_argument("--target-model-dir", required=True)
    p.add_argument("--grader", default="gemini-3.5-flash")
    p.add_argument("--name", required=True, help="run name under results/grade_impact/")
    p.add_argument("--students", default=None, help="comma list to restrict (e.g. the 8 eval students)")
    p.add_argument("--repeats", type=int, default=0, help="extra gold re-gradings for the noise floor")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()

    data = Path(a.data_root)
    items = pd.read_csv(a.items or data / "Screenshot_output_anon" / "set_splitting" / "obsetf_involved_data.csv", dtype=str)
    if a.students:
        items = items[items["Student ID"].isin(a.students.split(","))]
    cache = Cache(PROJECT_ROOT / "results" / "llm_cache.sqlite")

    rows = []
    for _, r in items.iterrows():
        hw, st, qid = r["Homework ID"], r["Student ID"], r["Question ID"]
        gold = load_markdown(data / RECTIFIED, hw, st, qid, "gemini-2.5-pro")  # expert-rectified only
        tgt = load_markdown(Path(a.target_root), hw, st, qid, a.target_model_dir)
        rp = rubric_path(data, qid)
        if gold is None or tgt is None or not rp.exists():
            continue
        rows.append((hw, st, qid, gold, tgt, json.loads(rp.read_text(encoding="utf-8")), r.get("gt_per_code")))
    if a.limit:
        rows = rows[: a.limit]
    print(f"{len(rows)} samples with expert transcript + target transcript + rubric")

    def work(x):
        hw, st, qid, gold, tgt, rub, gt = x
        g = grade(gold, rub, a.grader, cache)
        t = grade(tgt, rub, a.grader, cache)
        reps = [grade(gold, rub, a.grader, cache, nonce=str(k + 1)) for k in range(a.repeats)]
        return hw, st, qid, rub, gt, g, t, reps

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(tqdm(ex.map(work, rows), total=len(rows), desc="grading"))

    code_rows, sample_rows, failed = [], [], 0
    for hw, st, qid, rub, gt, g, t, reps in res:
        if g is None or t is None:
            failed += 1
            continue
        ta = parse_codes(gt)
        for c in g:
            code_rows.append({"Homework ID": hw, "Student ID": st, "Question ID": qid, "code": c,
                              "gold": g[c], "target": t[c], "ta": ta.get(c, 0.0),
                              **{f"gold_rep{k + 1}": (rr[c] if rr else np.nan) for k, rr in enumerate(reps)}})
        sample_rows.append({"Homework ID": hw, "Student ID": st, "Question ID": qid,
                            "score_gold": 1 - sum(g.values()), "score_target": 1 - sum(t.values()),
                            "score_ta": 1 - sum(ta.values())})
    cdf, sdf = pd.DataFrame(code_rows), pd.DataFrame(sample_rows)
    eps = 1e-9
    cdf["tce"] = (cdf.target - cdf.gold).abs() > eps
    cdf["wrongful"] = cdf.target > cdf.gold + eps
    cdf["overcredit"] = cdf.target < cdf.gold - eps
    samp = cdf.groupby(KEYS).agg(tce=("tce", "any"), wrongful=("wrongful", "any"), overcredit=("overcredit", "any")).reset_index()

    m = {
        "name": a.name, "grader": a.grader, "target": f"{a.target_root} :: {a.target_model_dir}",
        "n_samples": int(len(sdf)), "n_codes": int(len(cdf)), "grading_failures": failed,
        "code_level": {
            "TCE_rate_percent": round(100 * cdf.tce.mean(), 2),
            "wrongful_deduction_percent": round(100 * cdf.wrongful.mean(), 2),
            "over_credit_percent": round(100 * cdf.overcredit.mean(), 2),
            "TCE_95CI_student_bootstrap": boot_ci(cdf.tce.astype(float), cdf["Student ID"]),
        },
        "sample_level": {
            "TCE_rate_percent": round(100 * samp.tce.mean(), 2),
            "with_wrongful_deduction_percent": round(100 * samp.wrongful.mean(), 2),
            "with_over_credit_percent": round(100 * samp.overcredit.mean(), 2),
            "TCE_95CI_student_bootstrap": boot_ci(samp.tce.astype(float), samp["Student ID"]),
        },
        "score_MAE_target_vs_gold": round(float((sdf.score_target - sdf.score_gold).abs().mean()), 4),
        "grader_validity_gold_vs_TA": {
            "code_exact_agreement_percent": round(100 * float(((cdf.gold - cdf.ta).abs() < eps).mean()), 2),
            "score_MAE": round(float((sdf.score_gold - sdf.score_ta).abs().mean()), 4),
        },
    }
    rep_cols = [c for c in cdf.columns if c.startswith("gold_rep")]
    if rep_cols:
        flips = np.mean([((cdf[c] - cdf.gold).abs() > eps).mean() for c in rep_cols])
        m["grader_noise_floor_code_flip_percent"] = round(100 * float(flips), 2)
    out = PROJECT_ROOT / "results" / "grade_impact" / a.name
    out.mkdir(parents=True, exist_ok=True)
    cdf.to_csv(out / "per_code.csv", index=False)
    sdf.merge(samp, on=KEYS).to_csv(out / "per_sample.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
