"""Correction Inception on real exams: two agents negotiate a shared claim ledger, an evidence layer verifies,
and a rule-based Arbiter decides. Two protocols are implemented:

INTERACTIVE (the main system; file -> verify -> rebut -> decide). Each phase is a separate process, so exactly
one model is in memory at a time, and every phase resumes by re-running it:
  r1        ROUND 1: both agents file claims + tool requests, blind to each other
  tools     executes the round's tool requests (--round 1) -> results go into the shared ledger
  cross     ROUND 2: both agents see the FULL ledger (opponent claims + all evidence) and may
            new / withdraw / revise / counter / defend
  tools     executes round-2 tool requests (--round 2)
  rebut     ROUND 3: only where the two agents still propose different values for the same span;
            each must withdraw / concede / stand (a human reviews if both stand)
  decide    Arbiter (fixed rules) over final claims -> variants, ledger, interaction + tool stats

  Ledger files in <out>/<name>/: r1.jsonl, tools.jsonl, r2.jsonl, r3.jsonl (append-only, replayed by decide).

ONE-SHOT (the ablation baseline, kept for comparability):
  claims      LLM agents file claims for each exam, no tools, no interaction
  reread      faithful reader re-reads each claimed line with the disputed value HIDDEN
  decide      Arbiter -> variants + theta sweep + claim precision
  selfcorrect baseline B2: one LLM pass "fix the transcription errors"

Variants written by `decide` (same folder layout as the transcribers, so every scorer works unchanged):
  none           original transcript (re-scored in the same run, for a clean before/after)
  agents         full interactive system: negotiation + re-read + consistency + Arbiter
  noverify       all well-formed live claims accepted (ablation: no verification)
  advocate_only  / auditor_only   (ablation: one agent)
  selfcorrect    (only from the one-shot path) one-pass LLM repair

  python run_agents.py r1    --target-root R --model-dir M --name N --out OUT --llm hf-text:Qwen/Qwen2.5-7B-Instruct
  python run_agents.py tools --round 1 [same args] --images IMG --reader hf:... --adapter A
  python run_agents.py cross [same args]
  python run_agents.py tools --round 2 [same args]
  python run_agents.py rebut [same args]
  python run_agents.py decide [same args]

  ONE PROCESS, TWO GPUs (both models resident, exams pipelined):
  python run_agents.py auto  [same args] --agent-device 0 --reader-device 1 --concurrency 6
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
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

from agents import _parse_claims, agent_prompt, agent_prompt_r1, agent_prompt_r2, agent_prompt_r3  # noqa: E402
from arith_consistency import evaluate, latex_to_expr  # noqa: E402
from core import ACCEPT, ArbiterConfig, Claim, apply_accepted, arbitrate, changed_number, span_check  # noqa: E402
from protocol import (Ledger, ToolBox, apply_actions, build_ledger, conflict_groups,  # noqa: E402
                      ledger_json, parse_actions, parse_single_action, parse_tool_requests)
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


# ================================================================== interactive protocol
# File -> Verify -> Rebut -> Decide. Each phase is a separate process so exactly one model is in memory
# (7B agents, then 3B reader) and each phase resumes by re-running it.
def _ledger_recs(out: Path, exam: str, name: str) -> list:
    return [r for r in read_jsonl(out / f"{name}.jsonl").values() if r.get("exam") == exam]


def exam_ledger(out: Path, exam: str, transcript: str) -> Ledger:
    """Rebuild the shared ledger for one exam by replaying the JSONL phases (deterministic, same cids)."""
    return build_ledger(exam, transcript, _ledger_recs(out, exam, "r1"), _ledger_recs(out, exam, "tools"),
                        _ledger_recs(out, exam, "r2") + _ledger_recs(out, exam, "r3"))


def _resolve_for_claim(led, agent: str, rnd: int, req: dict) -> str | None:
    """Which claim is this tool request about? Accepts an explicit '#cid', an index, or a span."""
    fc = req.get("for_claim")
    if isinstance(fc, str) and fc.strip().startswith("#"):
        return fc.strip() if fc.strip() in led.claims else None
    own = [c for c in led.of(agent) if c.round_filed == rnd]
    if isinstance(fc, bool):
        fc = None
    if isinstance(fc, (int, float)) or (isinstance(fc, str) and fc.strip().isdigit()):
        k = int(fc)
        return own[k].cid if 0 <= k < len(own) else None
    if isinstance(fc, str) and fc.strip():
        for c in own:
            if fc.strip() in c.span:
                return c.cid
    return own[0].cid if len(own) == 1 else None


def make_reread(a, hw, st, qid, reader, idx):
    """crop_reread for one exam: find the claimed line, hide the disputed number, re-read the page image."""
    def reread(span, proposed):
        if a.reader == "mock":
            return proposed, 0.9
        page, line = page_of_span(Path(a.target_root), hw, st, qid, a.model_dir, span)
        if not page or not line:
            return "", 0.0
        masked = masked_line(line, span, proposed)
        img = idx.get((hw, st, page))
        if not masked or img is None:
            return "", 0.0
        from PIL import Image
        return reader.read_with_conf(REREAD_PROMPT.format(masked=masked), Image.open(img).convert("RGB"))
    return reread


def _exam_jobs(a):
    """(exam, hw, st, qid, transcript, rubric, hints) for every exam we can process."""
    jobs = []
    for hw, st, qid in items(a.limit):
        exam = key_of(hw, st, qid)
        t = load_markdown(Path(a.target_root), hw, st, qid, a.model_dir)
        rp = rubric_path(Path(a.edu_data), qid)
        if t is None or not rp.exists():
            continue
        rubric = json.loads(rp.read_text(encoding="utf-8"))
        jobs.append((exam, hw, st, qid, t, rubric, hints_for(t)))
    return jobs


def phase_r1(a):
    """ROUND 1: both agents file claims + tool requests, blind to each other."""
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    done = read_jsonl(out / "r1.jsonl")
    cache = Cache(Path(a.out) / "agents_llm_cache.sqlite") if a.llm != "mock" else None
    jobs = [(exam, agent, t, rubric, hints) for exam, hw, st, qid, t, rubric, hints in _exam_jobs(a)
            for agent in ("advocate", "auditor") if f"{exam}|{agent}" not in done]
    print(f"{len(jobs)} round-1 calls ({len(done)} done before)")

    def one(job):
        exam, agent, t, rubric, hints = job
        raw = call_llm(agent_prompt_r1(agent, t, rubric, hints, a.max_claims, a.tool_budget), a.llm,
                       temperature=0.0, cache=cache)
        return {"id": f"{exam}|{agent}", "exam": exam, "agent": agent, "round": 1, "raw": raw[:4000],
                "claims": parse_actions(raw), "tool_requests": parse_tool_requests(raw),
                "error": raw.startswith("Error:")}

    with open(out / "r1.jsonl", "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        for rec in tqdm(ex.map(one, jobs), total=len(jobs), desc=f"r1 {a.name}"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


def phase_tools(a):
    """Execute the tool requests made in round `--round`. One reader per process; results are logged."""
    out = Path(a.out) / a.name
    rnd = int(a.round or 1)
    src = {1: "r1", 2: "r2", 3: "r3"}[rnd]
    path = out / "tools.jsonl"
    done = read_jsonl(path)
    idx = page_index(Path(a.images)) if a.images else {}
    reader = None
    if a.reader != "mock":
        from run_ocr import HFReader
        reader = HFReader(a.reader[3:], a.adapter, 1280 * 28 * 28, 96)
    todo = []
    for rec in read_jsonl(out / f"{src}.jsonl").values():
        for i, req in enumerate(rec.get("tool_requests") or []):
            rid = f"{rec['id']}|{rnd}|{i}"
            if rid not in done:
                todo.append((rid, rec, req))
    print(f"{len(todo)} tool calls to run for round {rnd} ({len(done)} done before)")
    ledgers, boxes = {}, {}
    with open(path, "a", encoding="utf-8") as f:
        for rid, rec, req in tqdm(todo, desc=f"tools r{rnd} {a.name}"):
            exam, agent = rec["exam"], rec["agent"]
            hw, st, qid = exam.split("|")
            t = load_markdown(Path(a.target_root), hw, st, qid, a.model_dir) or ""
            if exam not in boxes:
                ledgers[exam] = exam_ledger(out, exam, t)
                rubric = json.loads((rubric_path(Path(a.edu_data), qid)).read_text(encoding="utf-8")) \
                    if rubric_path(Path(a.edu_data), qid).exists() else {"criteria": []}
                boxes[exam] = ToolBox(t, rubric, reread=make_reread(a, hw, st, qid, reader, idx), budget=a.tool_budget)
            cid = _resolve_for_claim(ledgers[exam], agent, rnd, req)
            res = boxes[exam].call(agent, str(req.get("tool", "")), req.get("args") or {})
            f.write(json.dumps({"id": rid, "exam": exam, "agent": agent, "round": rnd, "cid": cid, **res},
                               ensure_ascii=False) + "\n")
            f.flush()


def phase_cross(a):
    """ROUND 2: both agents see the full ledger (opponent claims + all evidence) and act on it."""
    out = Path(a.out) / a.name
    done = read_jsonl(out / "r2.jsonl")
    cache = Cache(Path(a.out) / "agents_llm_cache.sqlite") if a.llm != "mock" else None
    jobs = []
    for exam, hw, st, qid, t, rubric, hints in _exam_jobs(a):
        led = exam_ledger(out, exam, t)
        if not led.claims:
            continue  # nobody filed anything: nothing to cross-examine
        for agent in ("advocate", "auditor"):
            if f"{exam}|{agent}" not in done:
                jobs.append((exam, agent, t, rubric, hints, led.render()))
    print(f"{len(jobs)} round-2 calls ({len(done)} done before)")

    def one(job):
        exam, agent, t, rubric, hints, ledger_txt = job
        raw = call_llm(agent_prompt_r2(agent, t, rubric, ledger_txt, hints, a.tool_budget), a.llm,
                       temperature=0.0, cache=cache)
        return {"id": f"{exam}|{agent}", "exam": exam, "agent": agent, "round": 2, "raw": raw[:4000],
                "actions": parse_actions(raw), "tool_requests": parse_tool_requests(raw),
                "error": raw.startswith("Error:")}

    with open(out / "r2.jsonl", "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        for rec in tqdm(ex.map(one, jobs), total=len(jobs), desc=f"r2 {a.name}"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


def phase_rebut(a):
    """ROUND 3: only for spans where the two agents still propose different values."""
    out = Path(a.out) / a.name
    done = read_jsonl(out / "r3.jsonl")
    cache = Cache(Path(a.out) / "agents_llm_cache.sqlite") if a.llm != "mock" else None
    jobs = []
    for exam, hw, st, qid, t, rubric, hints in _exam_jobs(a):
        led = exam_ledger(out, exam, t)
        for group in conflict_groups(led):
            for cid in group:
                if f"{exam}|{cid}" not in done:
                    jobs.append((exam, cid, t, led, group))
    print(f"{len(jobs)} rebuttal calls ({len(done)} done before)")

    def one(job):
        exam, cid, t, led, group = job
        rec = led.claims[cid]
        raw = call_llm(agent_prompt_r3(rec.agent, t, rec.span, led.render(only=group), a.rebut_tool_budget), a.llm,
                       temperature=0.0, cache=cache)
        action = parse_single_action(raw) or {"action": "stand", "reason": "no valid round-3 response"}
        action.setdefault("cid", cid)
        return {"id": f"{exam}|{cid}", "exam": exam, "agent": rec.agent, "round": 3, "raw": raw[:2000],
                "actions": [action], "tool_requests": parse_tool_requests(raw), "error": raw.startswith("Error:")}

    with open(out / "r3.jsonl", "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        for rec in tqdm(ex.map(one, jobs), total=len(jobs), desc=f"r3 {a.name}"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


def decide_interactive(a, out: Path) -> dict:
    """Arbiter over the final claims of the interactive ledger -> variants, stats, precision."""
    var_root = out / "Outputs"
    ledger_rows, stats = [], Counter()
    interaction = Counter()
    tools = Counter()
    precision = {v: Counter() for v in ("all_claims", "agents", "noverify", "advocate_only", "auditor_only")}
    sweep = {th: Counter() for th in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)}

    def decide_group(claims, theta):
        return arbitrate(claims, t, lambda c: (c.evidence.get("reread", ""), float(c.evidence.get("reread_conf") or 0.0)),
                         ArbiterConfig(theta=theta), use_existing_evidence=True)

    for hw, st, qid in items(a.limit):
        exam = key_of(hw, st, qid)
        t = load_markdown(Path(a.target_root), hw, st, qid, a.model_dir)
        if t is None:
            continue
        gold = load_markdown(Path(a.edu_data) / RECTIFIED, hw, st, qid, "gemini-2.5-pro") or ""
        led = exam_ledger(out, exam, t)
        if not led.claims:
            continue
        stats["samples"] += 1
        before = len(conflict_groups(led))  # conflicts still standing after round 3
        variants = {"none": t}
        final = decide_group(led.final_claims(), a.theta)
        variants["agents"] = apply_accepted(t, final)
        noverify = [Claim(**{**c.to_claim().__dict__, "decision": ACCEPT}) for c in led.live()
                    if span_check(t, c.span) and changed_number(c.span, c.proposed) is not None]
        variants["noverify"] = apply_accepted(t, noverify)
        for name in ("advocate", "auditor"):
            sub = [c for c in led.final_claims() if c.agent == name]
            variants[f"{name}_only"] = apply_accepted(t, decide_group(sub, a.theta)) if sub else t
        for v, text in variants.items():
            write_variant(var_root, v, hw, st, qid, a.model_dir, text)
        for th in sweep:
            for c in decide_group(led.final_claims(), th):
                sweep[th][c.decision] += 1
        for c in final:
            stats[f"{c.agent}_{c.decision}"] += 1
            ledger_rows.append({"id": exam, **c.__dict__})
        for name, cs in (("all_claims", [c.to_claim() for c in led.live()]),
                         ("agents", [c for c in final if c.decision == ACCEPT]),
                         ("noverify", noverify),
                         ("advocate_only", [c for c in decide_group([x for x in led.final_claims() if x.agent == "advocate"], a.theta) if c.decision == ACCEPT]),
                         ("auditor_only", [c for c in decide_group([x for x in led.final_claims() if x.agent == "auditor"], a.theta) if c.decision == ACCEPT])):
            for c in cs:
                h = claim_helps(gold, t, c.span, c.proposed)
                precision[name]["helps" if h else ("hurts" if h is False else "neutral")] += 1
        # interaction + tool accounting from the ledger itself
        for rec in led.claims.values():
            interaction["filed_r1" if rec.round_filed == 1 else "filed_r2"] += 1
            if rec.superseded_by:
                interaction["revised"] += 1
            if rec.counters:
                interaction["countered"] += 1
            if rec.status == "withdrawn":
                interaction["withdrawn"] += 1
            if rec.status == "conceded":
                interaction["conceded"] += 1
            if rec.status == "final":
                interaction["stood"] += 1
            for h in rec.history:
                if h.get("action") == "defend":
                    interaction["defended"] += 1
        interaction["conflicts_after_r3"] += before
        interaction["rejected_actions"] += len(led.rejected)
        for res in led.tool_log:
            tools[res.get("tool", "?")] += 1
            tools["ok" if res.get("ok") else "failed"] += 1
        with open(out / "interactive_ledger.jsonl", "a", encoding="utf-8") as f:
            f.write(ledger_json(led).replace("\n", " ") + "\n")

    def pct(v: Counter, k: str) -> float:
        return round(100 * v[k] / max(1, v["helps"] + v["hurts"]), 1)

    summary = {
        "name": a.name, "protocol": "interactive (r1 -> tools -> r2 -> tools -> r3 -> decide)", "theta": a.theta,
        "tool_budget_per_agent": a.tool_budget,
        "counts": dict(stats),
        "interaction": dict(interaction),
        "tools": dict(tools),
        "theta_sweep": {str(k): dict(v) for k, v in sweep.items()},
        "claim_precision_vs_expert": {k: {**dict(v), "precision_percent": pct(v, "helps")} for k, v in precision.items()},
    }
    (out / "decide_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def phase_auto(a):
    """The whole negotiation per exam, with BOTH models resident: the agents on --agent-device and the
    faithful reader on --reader-device. Exams are processed concurrently, so the 7B batches across exams
    while the reader is verifying another exam's claims (the two GPUs work at the same time).

    Records for an exam are written only when that exam's pipeline finishes, so a crash mid-exam leaves
    nothing behind and re-running simply redoes that exam. Finished exams are skipped via auto_done.jsonl."""
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    if a.llm.startswith("hf-text:") and not os.environ.get("HF_TEXT_DEVICES"):
        os.environ["HF_TEXT_DEVICES"] = str(a.agent_device)   # keep the text model off the reader's GPU
    cache = Cache(Path(a.out) / "agents_llm_cache.sqlite") if a.llm != "mock" else None
    reader = None
    if a.reader != "mock":
        from run_ocr import HFReader
        reader = HFReader(a.reader[3:], a.adapter, 1280 * 28 * 28, 96, device=a.reader_device)
        print(f"reader on cuda:{a.reader_device}; agents on cuda:{a.agent_device}")
    reader_lock = threading.Lock()
    idx = page_index(Path(a.images)) if a.images else {}
    done = {r.get("exam") for r in read_jsonl(out / "auto_done.jsonl").values()}
    jobs = [j for j in _exam_jobs(a) if j[0] not in done]
    print(f"{len(jobs)} exams to negotiate ({len(done)} done before), concurrency {a.concurrency}")

    files = {k: open(out / f"{k}.jsonl", "a", encoding="utf-8") for k in ("r1", "tools", "r2", "r3", "auto_done")}
    lock = threading.Lock()

    def one(job):
        exam, hw, st, qid, t, rubric, hints = job
        recs = {"r1": [], "tools": [], "r2": [], "r3": []}

        def rebuild():
            return build_ledger(exam, t, recs["r1"], recs["tools"], recs["r2"] + recs["r3"])

        def run_tools(led, rnd, src_recs, default_cid=None):
            box = ToolBox(t, rubric, reread=make_reread(a, hw, st, qid, reader, idx), budget=a.tool_budget)
            for rec in src_recs:
                for i, req in enumerate(rec.get("tool_requests") or []):
                    cid = _resolve_for_claim(led, rec["agent"], rnd, req) or (default_cid if rnd == 3 else None)
                    try:
                        with reader_lock:
                            res = box.call(rec["agent"], str(req.get("tool", "")), req.get("args") or {})
                    except Exception as e:  # a tool fault must not lose the exam
                        res = {"tool": str(req.get("tool", "")), "ok": False, "error": f"{type(e).__name__}: {e}"}
                    recs["tools"].append({"id": f"{rec['id']}|{rnd}|{i}", "exam": exam, "agent": rec["agent"],
                                          "round": rnd, "cid": cid, **res})

        def ask(prompt):
            return call_llm(prompt, a.llm, temperature=0.0, cache=cache)

        # ---- round 1: independent filing
        for agent in ("advocate", "auditor"):
            raw = ask(agent_prompt_r1(agent, t, rubric, hints, a.max_claims, a.tool_budget))
            recs["r1"].append({"id": f"{exam}|{agent}", "exam": exam, "agent": agent, "round": 1, "raw": raw[:4000],
                               "claims": parse_actions(raw), "tool_requests": parse_tool_requests(raw),
                               "error": raw.startswith("Error:")})
        run_tools(rebuild(), 1, recs["r1"])
        led = rebuild()

        # ---- round 2: cross-examination on the shared ledger
        if led.claims:
            for agent in ("advocate", "auditor"):
                raw = ask(agent_prompt_r2(agent, t, rubric, led.render(), hints, a.tool_budget))
                recs["r2"].append({"id": f"{exam}|{agent}", "exam": exam, "agent": agent, "round": 2, "raw": raw[:4000],
                                   "actions": parse_actions(raw), "tool_requests": parse_tool_requests(raw),
                                   "error": raw.startswith("Error:")})
            run_tools(rebuild(), 2, recs["r2"])
            led = rebuild()

            # ---- round 3: rebuttal, only where the two agents still disagree
            for group in conflict_groups(led):
                for cid in group:
                    rec = led.claims[cid]
                    raw = ask(agent_prompt_r3(rec.agent, t, rec.span, led.render(only=group), a.rebut_tool_budget))
                    action = parse_single_action(raw) or {"action": "stand", "reason": "no valid round-3 response"}
                    action.setdefault("cid", cid)
                    recs["r3"].append({"id": f"{exam}|{cid}", "exam": exam, "agent": rec.agent, "round": 3,
                                       "raw": raw[:2000], "actions": [action],
                                       "tool_requests": parse_tool_requests(raw), "error": raw.startswith("Error:")})
                    run_tools(rebuild(), 3, [recs["r3"][-1]], default_cid=cid)

        with lock:  # one exam's records land together, so a crash never leaves a half-replayed ledger
            for k in ("r1", "tools", "r2", "r3"):
                for r in recs[k]:
                    files[k].write(json.dumps(r, ensure_ascii=False) + "\n")
                files[k].flush()
            files["auto_done"].write(json.dumps({"id": exam, "exam": exam}) + "\n")
            files["auto_done"].flush()
        return exam

    with ThreadPoolExecutor(a.concurrency) as ex:
        for _ in tqdm(ex.map(one, jobs), total=len(jobs), desc=f"auto {a.name}"):
            pass
    for f in files.values():
        f.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["claims", "reread", "decide", "selfcorrect",
                                     "r1", "tools", "cross", "rebut", "auto"])
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
    p.add_argument("--tool-budget", type=int, default=4, help="tool calls per agent per exam (all rounds)")
    p.add_argument("--rebut-tool-budget", type=int, default=1, help="tool calls allowed in the round-3 rebuttal")
    p.add_argument("--round", type=int, help="which round's tool requests to execute (tools phase)")
    p.add_argument("--agent-device", type=int, default=0, help="GPU for the agent LLM (auto phase)")
    p.add_argument("--reader-device", type=int, default=1, help="GPU for the faithful reader (auto phase)")
    p.add_argument("--concurrency", type=int, default=6, help="exams in flight at once (auto phase)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    if a.phase == "decide" and (Path(a.out) / a.name / "r1.jsonl").exists():
        decide_interactive(a, Path(a.out) / a.name)   # interactive ledger present -> interactive variants
        return
    {"claims": phase_claims, "reread": phase_reread, "decide": phase_decide, "selfcorrect": phase_selfcorrect,
     "r1": phase_r1, "tools": phase_tools, "cross": phase_cross, "rebut": phase_rebut,
     "auto": phase_auto}[a.phase](a)


if __name__ == "__main__":
    main()
