"""Free (no-API) baseline analysis from data shipped with EDU-CIRCUIT-HW.

Uses, for the 513 observation samples:
  * pred_per_code : the paper's LLM grader deductions (graded from the Gemini-2.5-Pro transcript)
  * gt_per_code   : TA-verified deductions (ground truth)
  * the paper's judge verdicts on the Gemini-2.5-Pro transcript (recognition errors found or not)

Questions:
  Q1. How often does the automatic grader disagree with the TA (per rubric code)? In which direction?
      over-credit = grader deducts LESS than TA; wrongful deduction = grader deducts MORE than TA.
  Q2. Is grading disagreement more frequent when the transcript contains recognition errors?
      (motivation for the project: reading errors propagate into grades)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "eval"))
from evaluate_recognition import DEFAULT_EDU_REPO, KEYS, SHIPPED, count_errors, load_env, read_csv_any  # noqa: E402


def codes(s) -> dict:
    try:
        return {k: float(v) for k, v in json.loads(s).items()} if isinstance(s, str) and s.strip() else {}
    except Exception:
        return {}


def boot(x: pd.Series, g: pd.Series, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    grp = [v.values for _, v in x.groupby(g)]
    s = [np.concatenate([grp[i] for i in rng.integers(0, len(grp), len(grp))]).mean() for _ in range(n)]
    return [round(100 * float(np.percentile(s, 2.5)), 1), round(100 * float(np.percentile(s, 97.5)), 1)]


def main():
    load_env(DEFAULT_EDU_REPO)
    data = Path(os.environ.get("EDU_DATA", "D:/lma/data/EDU-CIRCUIT-HW_v1"))
    items = pd.read_csv(data / "Screenshot_output_anon" / "set_splitting" / "obsetf_involved_data.csv", dtype=str)
    judge = read_csv_any(Path(DEFAULT_EDU_REPO) / SHIPPED / "0_finaloutput_rechecked" /
                         "Recognition_Detection_gemini-2.5-pro_obsetf_gemini-2.5-pro.csv")
    judge["rec_errors"] = judge["Recognition Errors"].apply(count_errors)
    df = items.merge(judge[KEYS + ["rec_errors"]], on=KEYS, how="left")

    rows = []
    for _, r in df.iterrows():
        rub = data / "Screenshot_output_anon" / "rubric_outputs" / f"P{r['Question ID'].replace('_', '.', 1)}.json"
        allc = [c["code"] for c in json.loads(rub.read_text(encoding="utf-8"))["criteria"]] if rub.exists() else []
        pr, gt = codes(r["pred_per_code"]), codes(r["gt_per_code"])
        for c in sorted(set(allc) | set(pr) | set(gt)):
            rows.append({**{k: r[k] for k in KEYS}, "code": c, "pred": pr.get(c, 0.0), "ta": gt.get(c, 0.0),
                         "rec_err": (r["rec_errors"] or 0) > 0})
    c = pd.DataFrame(rows)
    eps = 1e-9
    c["disagree"] = (c.pred - c.ta).abs() > eps
    c["overcredit"] = c.pred < c.ta - eps
    c["wrongful"] = c.pred > c.ta + eps
    s = c.groupby(KEYS).agg(disagree=("disagree", "any"), rec_err=("rec_err", "first"),
                            score_err=("pred", "sum")).reset_index()
    s["score_err"] = (c.groupby(KEYS)["pred"].sum().values - c.groupby(KEYS)["ta"].sum().values)

    def block(d, sd):
        return {"n_samples": int(len(sd)), "n_codes": int(len(d)),
                "code_disagreement_percent": round(100 * d.disagree.mean(), 2),
                "code_over_credit_percent": round(100 * d.overcredit.mean(), 2),
                "code_wrongful_deduction_percent": round(100 * d.wrongful.mean(), 2),
                "sample_disagreement_percent": round(100 * sd.disagree.mean(), 2),
                "sample_disagreement_95CI_student_bootstrap": boot(sd.disagree.astype(float), sd["Student ID"]),
                "score_MAE": round(float(sd.score_err.abs().mean()), 4)}

    res = {
        "all": block(c, s),
        "transcript_WITH_recognition_errors": block(c[c.rec_err], s[s.rec_err]),
        "transcript_WITHOUT_recognition_errors": block(c[~c.rec_err], s[~s.rec_err]),
    }
    out = ROOT / "results" / "baseline_reference" / "shipped_grading_analysis.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
