"""Model-agnostic re-creation of the EDU-CIRCUIT-HW recognition-error evaluation (steps 9-12).

Pipeline (identical prompts/parsing to the paper's released code; only orchestration differs):
  step 9  : LLM judge (default gemini-2.5-pro, T=0, one-shot example) compares each target
            transcript with the expert-verified reference and lists recognition errors.
  step 10 : LLM filter drops "too strict" error items (case-only, math-equivalent, ...).
  step 11 : metrics -- Sample Error Rate (SER, % of samples with >=1 error) and
            Average Error Count (AEC, errors per sample).
  step 12 : LLM classifies every error item into 4 categories
            (1 symbol/character, 2 structure/notation, 3 visual/diagram, 4 textual/logical flow).

Differences from their scripts (all deliberate, none change the metric definition):
  * no 20-sample debug cap (their step 9 stops after 20 items);
  * any transcriber can be evaluated, any judge can be plugged in;
  * API calls are cached and run in parallel; multi-part files are concatenated in sorted order;
  * we additionally report student-clustered bootstrap 95% CIs (marked "ours").

Usage:
  python evaluate_recognition.py --reference                      # paper numbers from shipped judge outputs (no API)
  python evaluate_recognition.py --self-test                      # plumbing test with a mock judge (no API)
  python evaluate_recognition.py --task repro_obs --model-dir gemini-2.5-pro     # evaluate a transcript run
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edu_compat import install_backend, load_edu_modules  # noqa: E402
from llm import Cache  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EDU_REPO = os.environ.get("EDU_REPO", str(PROJECT_ROOT / "external" / "CIRCUIT_EDU_HW_ACL"))
SHIPPED = Path("check_recognition_error_val") / "LLM-as-a-judge-selection"
ITEM_LIST = SHIPPED / "0_finaloutput_rechecked" / "Recognition_Detection_gemini-2.5-pro_obsetf_gemini-2.5-pro.csv"
FEWSHOT = SHIPPED / "fewshot_example_v2"
RECTIFIED = Path("Rectified_recognized_markdown_done_Anon") / "Final_4_LLM_judge"
ORIGINAL_LABEL = Path("Observationset_Final") / "v6_Gemini_2p5"
KEYS = ["Homework ID", "Student ID", "Question ID"]


# ------------------------------------------------------------------ helpers
def load_env(edu_repo):
    """Load API keys from ci_eval/.env, then the EDU repo's .env (never printed)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(Path(edu_repo) / ".env")


def read_csv_any(path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, dtype={k: str for k in KEYS})
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"cannot decode {path}")


def count_errors(text) -> int:
    """Verbatim logic of their step 11 `count_errors`."""
    if pd.isna(text) or "No significant errors found." in str(text):
        return 0
    if str(text).startswith("Error:"):
        return 1
    items = re.findall(r"^\d+\.\s*$", str(text), re.MULTILINE)
    if items:
        return len(items)
    return 1 if len(str(text).strip()) > 0 else 0


def load_markdown(base: Path, hw: str, st: str, qid: str, model_dir: str) -> str | None:
    """Same lookup as their step 9: {qid}(_[1-9])?_markdown.md, first line (image link) dropped."""
    folder = base / f"Homework_collected_database_trial_{hw}_{st}" / "models" / model_dir / "Compare"
    rx = re.compile(rf"^{re.escape(str(qid))}(_[1-9])?_markdown\.md$")
    files = sorted(f for f in glob.glob(str(folder / f"{qid}*_markdown.md")) if rx.match(os.path.basename(f)))
    if not files:
        return None
    parts = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            parts.append("\n".join(fh.read().split("\n")[1:]))
    return "".join(parts)


_NOSIG = "No significant errors found."
_ITEM_SPLIT = re.compile(r"\n+(?=\d+[\.\)]\s*(?:\n\s*)?Source:)")


def _squash(s: str) -> str:
    return re.sub(r"[\s$\\{}]", "", s)


def normalize_judge_output(text):
    """Make a weaker judge's output match the format the paper's steps 10-12 expect (ours, not the paper's).
    1) drop any preamble before the first numbered item ('**Your Judgement:**' etc.);
    2) drop no-op items whose Source and Rectified Version are identical;
    3) 'No significant errors found.' + explanation -> exactly that sentence.
    Gemini's outputs pass through unchanged."""
    if text is None or (isinstance(text, float) and pd.isna(text)) or str(text).startswith("Error:"):
        return text
    t = str(text).strip()
    m = re.search(r"^\s*1[\.\)]\s*(?:\n\s*)?Source:", t, re.MULTILINE)
    if not m:
        return _NOSIG if _NOSIG.lower()[:-1] in t.lower() else t
    kept = []
    for item in _ITEM_SPLIT.split(t[m.start():].strip()):
        parts = re.split(r"Rectified Version:|Reason:", item)
        if len(parts) >= 2 and _squash(re.sub(r"^\s*\d+[\.\)]\s*Source:", "", parts[0])) == _squash(parts[1]):
            continue  # no-op "error"
        item = item.split("\n" + _NOSIG.split(" ")[0])[0].strip()  # trailing "No significant errors ... rest"
        kept.append(re.sub(r"^\s*\d+", str(len(kept) + 1), item, count=1))
    return "\n\n".join(kept) if kept else _NOSIG


def _parse_list(x) -> list:
    if isinstance(x, list):
        return x
    try:
        v = ast.literal_eval(str(x))
        return v if isinstance(v, list) else []
    except Exception:
        return []


def compute_metrics(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    c = df["Recognition Errors"].apply(count_errors)
    out = {
        "n_samples": int(len(df)),
        "n_judge_errors": int(df["Recognition Errors"].astype(str).str.startswith("Error:").sum()),
        "SER_percent": round(100 * float((c > 0).mean()), 2),
        "AEC": round(float(c.mean()), 3),
        "total_errors": int(c.sum()),
    }
    cats = [f"Category {i}" for i in range(1, 5)]
    if all(k in df.columns for k in cats):
        counts = {k: int(df[k].apply(lambda x: len(_parse_list(x))).sum()) for k in cats}
        tot = sum(counts.values()) or 1
        out["taxonomy_counts"] = counts
        out["taxonomy_share_percent"] = {k: round(100 * v / tot, 1) for k, v in counts.items()}
    # ours: student-clustered bootstrap (resample students, keep all their samples)
    if "Student ID" in df.columns and n_boot:
        rng = np.random.default_rng(seed)
        groups = [g.values for _, g in c.groupby(df["Student ID"])]
        ser, aec = [], []
        for _ in range(n_boot):
            pick = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            ser.append(100 * (pick > 0).mean())
            aec.append(pick.mean())
        out["ours_student_bootstrap_95CI"] = {
            "n_students": len(groups),
            "SER_percent": [round(float(np.percentile(ser, 2.5)), 2), round(float(np.percentile(ser, 97.5)), 2)],
            "AEC": [round(float(np.percentile(aec, 2.5)), 3), round(float(np.percentile(aec, 97.5)), 3)],
        }
    return out


# ------------------------------------------------------------------ pipeline steps
def step9_judge(judge_mod, items: pd.DataFrame, transcripts_root: Path, model_dir: str,
                data_root: Path, fewshot: Path, workers: int) -> pd.DataFrame:
    def one(row):
        hw, st, qid = row["Homework ID"], row["Student ID"], row["Question ID"]
        label = load_markdown(data_root / RECTIFIED, hw, st, qid, "gemini-2.5-pro")
        if label is None:  # their fallback: expert did not rectify -> original transcript is the reference
            label = load_markdown(data_root / ORIGINAL_LABEL, hw, st, qid, "gemini-2.5-pro")
        if label is None:
            return {**dict(zip(KEYS, (hw, st, qid))), "Recognition Errors": None, "status": "missing_label"}
        target = load_markdown(transcripts_root, hw, st, qid, model_dir)
        if target is None:
            return {**dict(zip(KEYS, (hw, st, qid))), "Recognition Errors": None, "status": "missing_target"}
        out = judge_mod.judge_difference_v7(
            target_content=target, label_content=label, example_folder_path=str(fewshot)
        )
        return {**dict(zip(KEYS, (hw, st, qid))), "Recognition Errors": out, "status": "ok"}

    rows = [r for _, r in items.iterrows()]
    with ThreadPoolExecutor(workers) as ex:
        res = list(tqdm(ex.map(one, rows), total=len(rows), desc="step 9 judge"))
    return pd.DataFrame(res)


def step10_filter(step10_mod, df: pd.DataFrame, judge_model: str, workers: int) -> pd.DataFrame:
    def one(text):
        processed, changed = step10_mod.process_recognition_errors(text, judge_model)
        return processed if changed else text

    with ThreadPoolExecutor(workers) as ex:
        new = list(tqdm(ex.map(one, df["Recognition Errors"].tolist()), total=len(df), desc="step 10 filter"))
    df = df.copy()
    df["Recognition Errors"] = new
    return df


def step12_taxonomy(step12_mod, df: pd.DataFrame, judge_model: str, workers: int) -> pd.DataFrame:
    """Same logic as their classify_recognition_errors_pipeline.process_row, parallelised."""
    split_pattern = r"\n+(?=\d+[\.\)]\s*(?:\n\s*)?Source:)"

    def one(text):
        content = str(text).strip() if pd.notna(text) else ""
        buckets = {1: [], 2: [], 3: [], 4: []}
        if content.startswith("1."):
            item_id = 1
            for item in re.split(split_pattern, content):
                item = item.strip()
                if not item or "Source:" not in item:
                    continue
                m = re.search(r"[1-4]", step12_mod.LLM_classify_item(item, judge_model))
                if m:
                    buckets[int(m.group())].append(item_id)
                item_id += 1
        return [buckets[i] for i in range(1, 5)]

    with ThreadPoolExecutor(workers) as ex:
        res = list(tqdm(ex.map(one, df["Recognition Errors"].tolist()), total=len(df), desc="step 12 taxonomy"))
    df = df.copy()
    for i in range(4):
        df[f"Category {i + 1}"] = [r[i] for r in res]
    return df


def run(args) -> dict:
    edu = Path(args.edu_repo)
    data_root = Path(args.data_root or edu)
    out_dir = Path(args.out) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    judge_mod, step10_mod, step12_mod = load_edu_modules(edu)
    cache = Cache(Path(args.out) / "llm_cache.sqlite") if args.judge != "mock" else None
    install_backend([judge_mod, step10_mod, step12_mod], args.judge, cache)

    items = read_csv_any(args.items or (edu / ITEM_LIST))[KEYS]
    if args.limit:
        items = items.head(args.limit)
    transcripts_root = Path(args.transcripts_root or (data_root / "Outputs")) / args.task

    s9 = step9_judge(judge_mod, items, transcripts_root, args.model_dir, data_root, edu / FEWSHOT, args.workers)
    s9.to_csv(out_dir / "step9_original.csv", index=False, encoding="utf-8-sig")
    status = s9["status"].value_counts().to_dict()
    ok = s9[s9["status"] == "ok"].drop(columns="status")
    if ok.empty:
        raise SystemExit(f"No samples judged. Status counts: {status}")
    if "gemini" not in args.judge.lower() and not args.no_normalize:
        ok = ok.assign(**{"Recognition Errors": ok["Recognition Errors"].apply(normalize_judge_output)})

    s10 = step10_filter(step10_mod, ok, args.judge, args.workers)
    s10.to_csv(out_dir / "step10_rechecked.csv", index=False, encoding="utf-8-sig")
    final = s10
    if not args.skip_taxonomy:
        final = step12_taxonomy(step12_mod, s10, args.judge, args.workers)
        final.to_csv(out_dir / "step12_taxonomy.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "task": args.task, "transcriber": args.model_dir, "judge": args.judge,
        "sample_status": status,
        "metrics_before_filter_step9": compute_metrics(ok, n_boot=0),
        "metrics_final": compute_metrics(final),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


def reference(args):
    """Recompute the paper's table from the judge outputs shipped in their repo (no API)."""
    edu = Path(args.edu_repo)
    rows = []
    for f in sorted((edu / SHIPPED / "0_finaloutput_rechecked").glob("Recognition_Detection_*_obsetf_gemini-2.5-pro.csv")):
        model = f.name[len("Recognition_Detection_"):-len("_obsetf_gemini-2.5-pro.csv")]
        m = compute_metrics(read_csv_any(f))
        tax = edu / SHIPPED / "0_finaloutput_rechecked_taxonomy" / f.name
        if tax.exists():
            m.update({k: v for k, v in compute_metrics(read_csv_any(tax), n_boot=0).items() if k.startswith("taxonomy")})
        rows.append({"transcriber": model, **m})
    for r in rows:
        ci = r.get("ours_student_bootstrap_95CI", {})
        share = r.get("taxonomy_share_percent", {})
        print(f"{r['transcriber']:32s} n={r['n_samples']:4d}  SER={r['SER_percent']:5.1f}% "
              f"{ci.get('SER_percent', '')}  AEC={r['AEC']:.2f}  taxonomy%={list(share.values())}")
    out = Path(args.out) / "reference_paper_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {out}")


def self_test(args):
    """Build a 3-sample fake layout and run the full pipeline with the mock judge."""
    tmp = Path(tempfile.mkdtemp(prefix="ci_eval_selftest_"))
    try:
        items = pd.DataFrame([["Homework1", "student_1", "1_5-2"],
                              ["Homework1", "student_1", "1_5-3"],
                              ["Homework1", "student_2", "1_5-2"]], columns=KEYS)
        items.to_csv(tmp / "items.csv", index=False)
        for i, (hw, st, qid) in enumerate(items.values):
            for base, model in [(tmp / RECTIFIED, "gemini-2.5-pro"), (tmp / "Outputs" / "selftest", "mymodel")]:
                d = base / f"Homework_collected_database_trial_{hw}_{st}" / "models" / model / "Compare"
                d.mkdir(parents=True, exist_ok=True)
                body = "I = 3 A" if base.name != "selftest" else ("MOCK_CLEAN" if i == 1 else "I = 8 A")
                (d / f"{qid}_markdown.md").write_text("![Source Image](x.png)\n" + body, encoding="utf-8")
        ns = argparse.Namespace(edu_repo=args.edu_repo, data_root=str(tmp), out=str(tmp / "results"),
                                task="selftest", model_dir="mymodel", judge="mock", items=str(tmp / "items.csv"),
                                limit=None, transcripts_root=None, workers=2, skip_taxonomy=False, no_normalize=False)
        m = run(ns)["metrics_final"]
        assert m["n_samples"] == 3 and m["total_errors"] == 2, m
        assert abs(m["SER_percent"] - 66.67) < 0.01 and m["taxonomy_counts"]["Category 1"] == 2, m
        print("\nSELF-TEST PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    load_env(DEFAULT_EDU_REPO)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--edu-repo", default=DEFAULT_EDU_REPO)
    p.add_argument("--data-root", default=os.environ.get("EDU_DATA"), help="where the Digital Library zip folders live (default: edu repo)")
    p.add_argument("--task", default="repro_obs", help="transcript run name (folder under Outputs/)")
    p.add_argument("--model-dir", default="gemini-2.5-pro", help="models/<name> folder of the transcriber")
    p.add_argument("--transcripts-root", default=None, help="default: <data-root>/Outputs")
    p.add_argument("--judge", default="gemini-2.5-pro", help="judge model (paper: gemini-2.5-pro)")
    p.add_argument("--items", default=None, help="CSV of samples to evaluate (default: the paper's 513 observation items)")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N samples (smoke test)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--skip-taxonomy", action="store_true")
    p.add_argument("--no-normalize", action="store_true", help="keep non-Gemini judge outputs raw")
    p.add_argument("--out", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--reference", action="store_true", help="recompute paper metrics from shipped outputs")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.reference:
        reference(args)
    elif args.self_test:
        self_test(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
