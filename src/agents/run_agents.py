"""Correction Inception on real exams: Advocate + Auditor claims -> context-free re-read -> rule-based Arbiter.

Phases (separate processes, so each model gets the whole GPU; every phase is resumable):
  claims     LLM agents file claims for each exam (hints: lines whose arithmetic does not add up)
  reread     faithful reader re-reads each claimed line from the page image, disputed value HIDDEN (context-free)
  decide     Arbiter (fixed rules) -> corrected transcripts for each variant + ledger + claim precision vs expert
  selfcorrect  baseline B2: one LLM pass "fix the transcription errors" (no claims, no evidence)

Variants written by `decide` (same folder layout as the transcribers, so every scorer works unchanged):
  none           original transcript (re-scored in the same run, for a clean before/after)
  agents         full system: both agents + re-read + consistency + Arbiter
  noverify       both agents, every well-formed claim accepted (ablation: no verification)
  advocate_only  / auditor_only   (ablation: one agent)

  python run_agents.py claims --target-root R --model-dir M --name N --out OUT --llm hf-text:Qwen/Qwen2.5-7B-Instruct
  python run_agents.py reread --target-root R --model-dir M --name N --out OUT --images IMG --reader hf:... --adapter A
  python run_agents.py decide --target-root R --model-dir M --name N --out OUT
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from agents import _parse_claims, agent_prompt  # noqa: E402
from arith_consistency import evaluate, latex_to_expr  # noqa: E402
from core import ACCEPT, ArbiterConfig, Claim, apply_accepted, arbitrate, changed_number, span_check  # noqa: E402
from evaluate_recognition import RECTIFIED, load_markdown  # noqa: E402
from grade_impact import rubric_path  # noqa: E402
from llm import Cache, call_llm  # noqa: E402

ITEMS = ROOT / "src" / "eval" / "items_with_expert_transcript.csv"
VARIANTS = ["none", "agents", "noverify", "advocate_only", "auditor_only"]
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def key_of(hw, st, qid):
    return f"{hw}|{st}|{qid}"


def items(limit=None):
    df = pd.read_csv(ITEMS, dtype=str)
    rows = [tuple(r) for r in df[["Homework ID", "Student ID", "Question ID"]].itertuples(index=False)]
    return rows[:limit] if limit else rows


def read_jsonl(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


# ------------------------------------------------------------------ hints (deterministic spot finder)
def hints_for(transcript: str, max_hints: int = 8) -> str:
    """Lines containing a purely numeric equality that does not hold (2% tolerance).
    Could be a misread OR a genuine student slip: the agents decide, the Arbiter verifies."""
    out = []
    for line in transcript.splitlines():
        segs = re.findall(r"\$\$?(.+?)\$\$?", line) or [line]
        for seg in segs:
            parts = re.split(r"(?<![<>!])=(?!=)", seg)
            for a, b in zip(parts, parts[1:]):
                ea, eb = latex_to_expr(a), latex_to_expr(b)
                if not ea or not eb or not re.search(r"[+\-*/]", ea.lstrip("-")):
                    continue
                va, vb = evaluate(ea), evaluate(eb)
                if va is None or vb is None or abs(va - vb) <= max(0.02 * abs(vb), 1e-6):
                    continue
                out.append(f"- `{line.strip()[:160]}`: left side evaluates to {va:g}, right side is {vb:g}")
    return "\n".join(list(dict.fromkeys(out))[:max_hints])  # de-duplicated, capped


# ------------------------------------------------------------------ phase 1: claims
def phase_claims(a):
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "claims.jsonl"
    done = read_jsonl(path)
    cache = Cache(Path(a.out) / "agents_llm_cache.sqlite") if a.llm != "mock" else None
    jobs = []
    for hw, st, qid in items(a.limit):
        t = load_markdown(Path(a.target_root), hw, st, qid, a.model_dir)
        rp = rubric_path(Path(a.edu_data), qid)
        if t is None or not rp.exists():
            continue
        rubric = json.loads(rp.read_text(encoding="utf-8"))
        hints = hints_for(t)
        for agent in ("advocate", "auditor"):
            cid = f"{key_of(hw, st, qid)}|{agent}"
            if cid not in done:
                jobs.append((cid, agent, t, rubric, hints))

    def one(job):
        cid, agent, t, rubric, hints = job
        raw = call_llm(agent_prompt(agent, t, rubric, hints, a.max_claims), a.llm, temperature=0.0, cache=cache)
        claims = [] if raw.startswith("Error:") else _parse_claims(raw, agent)[: a.max_claims]
        return {"id": cid, "agent": agent, "n_hints": hints.count("\n- ") + bool(hints), "error": raw.startswith("Error:"),
                "raw": raw[:2000], "claims": [c.__dict__ for c in claims]}

    print(f"{len(jobs)} agent calls to make ({len(done)} done before)")
    with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        for rec in tqdm(ex.map(one, jobs), total=len(jobs), desc=f"claims {a.name}"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


# ------------------------------------------------------------------ phase 2: context-free re-read
def page_index(images_root: Path):
    """(hw, st, sanitized page name) -> image path, same naming as run_ocr."""
    idx = {}
    for img in (images_root / "Screenshot_output_anon").glob("Homework*/student_*/*.png"):
        san = img.name.replace(".", "_").replace("(", "").replace(")", "").replace("_png", "")
        idx[(img.parent.parent.name, img.parent.name, san)] = img
    return idx


def page_of_span(target_root: Path, hw, st, qid, model_dir, span):
    """Which page (per-page markdown file) contains the span -> its sanitized page name, and the line."""
    folder = target_root / f"Homework_collected_database_trial_{hw}_{st}" / "models" / model_dir / "Compare"
    rx = re.compile(rf"^{re.escape(qid)}(_[1-9])?_markdown\.md$")
    sq = re.sub(r"\s+", "", span)
    for f in sorted(folder.glob(f"{qid}*_markdown.md")):
        if not rx.match(f.name):
            continue
        for line in f.read_text(encoding="utf-8").splitlines()[1:]:
            if sq and sq in re.sub(r"\s+", "", line):
                return f.name[: -len("_markdown.md")], line.strip()
    return None, None


def masked_line(line: str, span: str, proposed: str) -> str | None:
    """The claimed line with the disputed number replaced by '?' (so the re-reader cannot copy it)."""
    ch = changed_number(span, proposed)
    if ch is None:
        return None
    pat = rf"(?<![\d.]){re.escape(ch[0])}(?![\d.])"
    if span in line:
        return line.replace(span, re.sub(pat, "?", span, count=1), 1)
    return re.sub(pat, "?", line, count=1)


REREAD_PROMPT = ("Find the handwritten line on this page that matches this partial transcription, where ? hides one "
                 "value:\n{masked}\n\nWrite that single line exactly as the student wrote it, including the hidden "
                 "value. Copy any mistakes. Output only the line.")


def resolve(reading: str, masked: str, span: str, proposed: str) -> str:
    """Map the free-form re-read onto the Arbiter's question: did the student write `proposed` or `span`?"""
    ch = changed_number(span, proposed)
    if ch is None:
        return reading
    left = Counter(_NUM.findall(reading)) - Counter(_NUM.findall(masked))  # values the reader filled in
    has_new, has_old = left[ch[1]] > 0, left[ch[0]] > 0
    if has_new and not has_old:
        return proposed
    if has_old and not has_new:
        return span
    return reading


def phase_reread(a):
    from PIL import Image

    out = Path(a.out) / a.name
    claims = read_jsonl(out / "claims.jsonl")
    path = out / "rereads.jsonl"
    done = read_jsonl(path)
    idx = page_index(Path(a.images))
    todo = []
    for cid, rec in claims.items():
        hw, st, qid, _ = cid.split("|")
        t = load_markdown(Path(a.target_root), hw, st, qid, a.model_dir) or ""
        for i, c in enumerate(rec["claims"]):
            rid = f"{cid}|{i}"
            if rid in done or not span_check(t, c["span"]):
                continue
            todo.append((rid, hw, st, qid, c))
    reader = None
    if a.reader != "mock":
        from run_ocr import HFReader
        reader = HFReader(a.reader[3:], a.adapter, 1280 * 28 * 28, 96)
    print(f"{len(todo)} claims to re-read ({len(done)} done before)")
    deadline = float(os.environ.get("LMA_DEADLINE", "0") or 0)
    with open(path, "a", encoding="utf-8") as f:
        for rid, hw, st, qid, c in tqdm(todo, desc=f"reread {a.name}"):
            if deadline and time.time() > deadline:
                print("time budget reached: stopping re-reads (re-run to continue)", file=sys.stderr)
                break
            page, line = page_of_span(Path(a.target_root), hw, st, qid, a.model_dir, c["span"])
            masked = masked_line(line, c["span"], c["proposed"]) if line else None
            img = idx.get((hw, st, page)) if page else None
            rec = {"id": rid, "page": page, "masked": masked, "reading": "", "conf": 0.0, "resolved": ""}
            if masked and img is not None:
                if reader is None:  # mock: pretend the page shows the proposed value
                    reading, conf = masked.replace("?", changed_number(c["span"], c["proposed"])[1], 1), 0.9
                else:
                    reading, conf = reader.read_with_conf(REREAD_PROMPT.format(masked=masked),
                                                          Image.open(img).convert("RGB"))
                rec.update(reading=reading, conf=round(conf, 4),
                           resolved=resolve(reading, masked, c["span"], c["proposed"]))
            else:
                rec["note"] = "no single-number span / page not found: unverifiable"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


# ------------------------------------------------------------------ phase 3: arbiter + variants + claim precision
def claim_helps(gold: str, target: str, span: str, proposed: str) -> bool | None:
    """Ground truth for one claim (uses the expert transcript, ONLY for evaluation):
    True if the change moves the transcript's numbers toward the expert's, False if away, None if neutral."""
    ch = changed_number(span, proposed)
    if ch is None:
        return None
    nums = lambda s: Counter(float(x.lstrip("-")) for x in _NUM.findall(s))  # noqa: E731
    ng, nt = nums(gold), nums(target)
    o, n = float(ch[0].lstrip("-")), float(ch[1].lstrip("-"))
    before = abs(ng[o] - nt[o]) + abs(ng[n] - nt[n])
    after = abs(ng[o] - (nt[o] - 1)) + abs(ng[n] - (nt[n] + 1))
    return None if after == before else after < before


def write_variant(out_root: Path, variant: str, hw, st, qid, model_dir, text: str):
    d = out_root / variant / f"Homework_collected_database_trial_{hw}_{st}" / "models" / model_dir / "Compare"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{qid}_markdown.md").write_text(f"![Source Image]({qid}_source.png)\n" + text, encoding="utf-8")


def phase_decide(a):
    out = Path(a.out) / a.name
    claims = read_jsonl(out / "claims.jsonl")
    rereads = read_jsonl(out / "rereads.jsonl")
    var_root = out / "Outputs"
    ledger, stats = [], Counter()
    sweep = {th: Counter() for th in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)}
    precision = {v: Counter() for v in ("all_claims", "agents", "noverify", "advocate_only", "auditor_only")}
    for hw, st, qid in items(a.limit):
        t = load_markdown(Path(a.target_root), hw, st, qid, a.model_dir)
        if t is None:
            continue
        gold = load_markdown(Path(a.edu_data) / RECTIFIED, hw, st, qid, "gemini-2.5-pro") or ""
        mk = lambda c, agent, i: Claim(**{**{k: v for k, v in c.items() if k in Claim.__dataclass_fields__},  # noqa: E731
                                          "agent": agent, "evidence": {"rid": f"{key_of(hw, st, qid)}|{agent}|{i}"}})
        per_agent = {ag: [mk(c, ag, i) for i, c in enumerate(claims.get(f"{key_of(hw, st, qid)}|{ag}", {}).get("claims", []))]
                     for ag in ("advocate", "auditor")}
        stats["samples"] += 1

        def reread_fn(c):
            r = rereads.get(c.evidence.get("rid", ""), {})
            return (r.get("resolved") or r.get("reading", ""), float(r.get("conf", 0.0)))

        def decided(cs, theta=a.theta):
            fresh = [Claim(**{**c.__dict__, "evidence": dict(c.evidence), "decision": None}) for c in cs]
            rids = [c.evidence["rid"] for c in fresh]
            for c, rid in zip(fresh, rids):  # arbitrate() overwrites evidence; keep the re-read lookup key
                c.evidence = {"rid": rid}
            return arbitrate(fresh, t, lambda c: reread_fn(c), ArbiterConfig(theta=theta))

        both = per_agent["advocate"] + per_agent["auditor"]
        variants = {"none": t}
        full = decided(both)
        variants["agents"] = apply_accepted(t, full)
        nv = [Claim(**{**c.__dict__, "decision": ACCEPT}) for c in both
              if span_check(t, c.span) and changed_number(c.span, c.proposed) is not None]
        variants["noverify"] = apply_accepted(t, nv)
        adv, aud = decided(per_agent["advocate"]), decided(per_agent["auditor"])
        variants["advocate_only"] = apply_accepted(t, adv)
        variants["auditor_only"] = apply_accepted(t, aud)
        for v, text in variants.items():
            write_variant(var_root, v, hw, st, qid, a.model_dir, text)
        for th, cnt in sweep.items():
            for c in decided(both, th):
                cnt[c.decision] += 1
        for c in full:
            stats[f"{c.agent}_claims"] += 1
            stats[f"{c.agent}_{c.decision}"] += 1
            ledger.append({"id": key_of(hw, st, qid), **c.__dict__})
        for name, cs in (("all_claims", both), ("agents", [c for c in full if c.decision == ACCEPT]), ("noverify", nv),
                         ("advocate_only", [c for c in adv if c.decision == ACCEPT]),
                         ("auditor_only", [c for c in aud if c.decision == ACCEPT])):
            for c in cs:
                h = claim_helps(gold, t, c.span, c.proposed)
                precision[name]["helps" if h else ("hurts" if h is False else "neutral")] += 1
    (out / "ledger.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in ledger), encoding="utf-8")
    summary = {
        "name": a.name, "theta": a.theta, "counts": dict(stats),
        "theta_sweep": {str(k): dict(v) for k, v in sweep.items()},
        "claim_precision_vs_expert": {k: {**dict(v), "precision_percent": round(100 * v["helps"] / max(1, v["helps"] + v["hurts"]), 1)}
                                      for k, v in precision.items()},
    }
    (out / "decide_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# ------------------------------------------------------------------ baseline B2: one-pass self-correction
SELF_PROMPT = """The following is an automatic transcription of a student's handwritten circuit-analysis answer.
It may contain transcription errors (misread digits, signs, subscripts). Return the transcript with transcription
errors fixed. Keep everything else identical. Output only the transcript.

TRANSCRIPT:
{transcript}"""


def phase_selfcorrect(a):
    out = Path(a.out) / a.name
    cache = Cache(Path(a.out) / "agents_llm_cache.sqlite") if a.llm != "mock" else None
    jobs = [(hw, st, qid, load_markdown(Path(a.target_root), hw, st, qid, a.model_dir)) for hw, st, qid in items(a.limit)]
    jobs = [j for j in jobs if j[3] is not None]

    def one(j):
        hw, st, qid, t = j
        d = out / "Outputs" / "selfcorrect" / f"Homework_collected_database_trial_{hw}_{st}" / "models" / a.model_dir / "Compare"
        if (d / f"{qid}_markdown.md").exists():
            return
        r = call_llm(SELF_PROMPT.format(transcript=t), a.llm, temperature=0.0, cache=cache)
        write_variant(out / "Outputs", "selfcorrect", hw, st, qid, a.model_dir, t if r.startswith("Error:") or not r.strip() else r)

    with ThreadPoolExecutor(a.workers) as ex:
        list(tqdm(ex.map(one, jobs), total=len(jobs), desc=f"selfcorrect {a.name}"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["claims", "reread", "decide", "selfcorrect"])
    p.add_argument("--target-root", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--edu-data", default=os.environ.get("EDU_DATA"))
    p.add_argument("--images")
    p.add_argument("--llm", default="hf-text:Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--reader", default="hf:Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter")
    p.add_argument("--theta", type=float, default=0.85)
    p.add_argument("--max-claims", type=int, default=3)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    {"claims": phase_claims, "reread": phase_reread, "decide": phase_decide, "selfcorrect": phase_selfcorrect}[a.phase](a)


if __name__ == "__main__":
    main()
