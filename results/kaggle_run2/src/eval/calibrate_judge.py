"""Calibrate a replacement judge against the paper's shipped Gemini-2.5-Pro judge verdicts.

Why: gemini-2.5-pro (the paper's judge) is no longer available to new API users, so we judge with
another model. This script measures how closely our judge reproduces the paper's verdicts on the
SAME transcripts (the paper's Gemini-2.5-Pro transcriptions, shipped in the dataset).

Usage (after running evaluate_recognition.py on the shipped transcripts):
  python evaluate_recognition.py --transcripts-root <EDU_DATA>/Observationset_Final --task v6_Gemini_2p5 \
         --model-dir gemini-2.5-pro --judge gemini-3.5-flash --out ../../results/judge_calibration
  python calibrate_judge.py --ours ../../results/judge_calibration/v6_Gemini_2p5/step10_rechecked.csv \
         --shipped gemini-2.5-pro
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_recognition import DEFAULT_EDU_REPO, KEYS, SHIPPED, count_errors, read_csv_any  # noqa: E402


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ours", required=True, help="our step10_rechecked.csv (or step12)")
    p.add_argument("--shipped", default="gemini-2.5-pro", help="transcriber name of the shipped judge file")
    p.add_argument("--edu-repo", default=DEFAULT_EDU_REPO)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    shipped = Path(a.edu_repo) / SHIPPED / "0_finaloutput_rechecked" / f"Recognition_Detection_{a.shipped}_obsetf_gemini-2.5-pro.csv"
    ours, ref = read_csv_any(a.ours), read_csv_any(shipped)
    m = ours.merge(ref, on=KEYS, suffixes=("_ours", "_paper"))
    co, cp = m["Recognition Errors_ours"].apply(count_errors), m["Recognition Errors_paper"].apply(count_errors)
    res = {
        "n_matched": int(len(m)),
        "SER_ours": round(100 * float((co > 0).mean()), 2), "SER_paper_judge": round(100 * float((cp > 0).mean()), 2),
        "AEC_ours": round(float(co.mean()), 3), "AEC_paper_judge": round(float(cp.mean()), 3),
        "sample_agreement_percent": round(100 * float(((co > 0) == (cp > 0)).mean()), 2),
        "cohen_kappa_has_error": round(cohen_kappa(co > 0, cp > 0), 3),
        "error_count_pearson_r": round(float(np.corrcoef(co, cp)[0, 1]), 3) if co.std() and cp.std() else None,
        "ours_flags_paper_clean": int(((co > 0) & (cp == 0)).sum()),
        "paper_flags_ours_clean": int(((cp > 0) & (co == 0)).sum()),
    }
    print(json.dumps(res, indent=2))
    out = Path(a.out or Path(a.ours).parent / "calibration_vs_paper_judge.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
