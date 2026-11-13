"""Advocate and Auditor as tool-calling LLM agents over a ledger of flagged numbers (C1)

The first agent run (run_agents.py) let the agents quote and rewrite transcript text: 623 claims changed more than one
number, 205 quoted spans that were not in the transcript, and none was accepted. Here the agents never touch text:

  ledger    per page, the numbers the pipeline flagged, each with an ID, its line (the number masked as [?]), the
            written value and the candidate values (beams + digit candidates)
  Advocate  (misreads that cost the student marks) round 1: calls tools by ID; round 2: files claims
            {"id", "change_to", "evidence"} with at most one claim per ID
  Auditor   (silent corrections / unsupported changes) round 1: calls tools on each claim; round 2: "support" or
            "object" per claim, with the tool result it relies on
  tools     answered from the evidence the pipeline already computed (no GPU at agent time):
              checker(id, value)     P(the student wrote `value`) from the fine-tuned A/B checker, both orders
              beams(id)              the reader's top whole numbers at that spot, with probabilities
              page_count(id, value)  how often `value` appears elsewhere on the page (the student's own work)
              blank(id, value)       the checker's preference WITHOUT the page (guessing from context) [Auditor]
              answer_key(id, value)  is `value` in the rubric's answer key (silent-correction risk) [Auditor]
            a value without checker evidence cannot be accepted
  Arbiter   a fixed rule, not an LLM: a claim changes the transcript only if the Auditor does not object and the
            checker conditions of correct.py hold at a threshold tuned on OTHER students (<= 1% false changes)

Variants, all on the same unseen students:
  pipeline        correct.py's rule, no agents (the baseline the agents must beat)
  agents          claims not objected to; threshold tuned on the other students' agent-approved numbers
  agents_filter   the same, but with the pipeline's own threshold (agents can only remove changes)
  advocate_only   claims without the Auditor's review

  python ledger_agents.py --dir results/checker/correct_checker_fe_k10_beams --target-root R --edu-data E \\
      --llm hf-text:Qwen/Qwen2.5-7B-Instruct --workers 16
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "detect"))
sys.path.insert(0, str(ROOT / "src" / "eval"))
from choice_prompt import masked_line  # noqa: E402
from evaluate_detection import _num  # noqa: E402

TOOL_BUDGET = 8
ADVOCATE_TOOLS = ("checker", "beams", "page_count")
AUDITOR_TOOLS = ("checker", "page_count", "blank", "answer_key")

ROLE = {
    "advocate": ("ROLE: ADVOCATE. You protect the student from MISREADS: the transcriber wrote a different number than "
                 "the student did, which can cost the student marks. Find flagged numbers that were misread and say "
                 "what the student actually wrote."),
    "auditor": ("ROLE: AUDITOR. You protect the grade from unearned marks and from wrong changes. Check each claim of the "
                "Advocate: a change is only right if the student really wrote that value. Object when the evidence "
                "suggests the value is guessed from context, is merely the expected answer, or the checker does not "
                "clearly prefer it."),
}
RULES = ("Rules: never judge whether the maths is correct (students make mistakes and those must be kept); only what is "
         "written counts. Refer to numbers by their ID only. Answer with ONE JSON object and nothing else.")


def parse_json(text: str):
    """First JSON object in a model reply, or None."""
    for m in re.finditer(r"\{", text or ""):
        depth = 0
        for k in range(m.start(), len(text)):
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            if depth == 0:
                try:
                    return json.loads(text[m.start():k + 1])
                except json.JSONDecodeError:
                    break
    return None


class Evidence:
    """The tool answers for one page, from verify*.csv / beams / page / rubric key."""

    def __init__(self, rows: pd.DataFrame, answer: str, key: set, beam_p: dict):
        self.rows = {k: r for k, (_, r) in enumerate(rows.iterrows())}
        self.answer, self.key, self.beam_p = answer, key, beam_p
        from consistency import page_numbers
        self.nums = page_numbers(answer)

    def entry(self, k: int) -> dict:
        r = self.rows[k]
        cands = [c for c in str(r["cands"] if pd.notna(r["cands"]) else "").split("|") if c]
        return {"id": k, "line": masked_line(self.answer, int(r["start"]), int(r["end"])), "written": str(r["text"]),
                "candidates": cands}

    def _p(self, r, value) -> float | None:
        for part in str(r.get("p_cands", "") if pd.notna(r.get("p_cands", "")) else "").split("|"):
            if ":" in part:
                c, p = part.rsplit(":", 1)
                if _num(c) is not None and _num(c) == _num(str(value)):
                    return float(p)
        return None

    def call(self, tool: str, k, value=None) -> dict:
        try:
            k = int(k)
        except (TypeError, ValueError):
            return {"error": "unknown id"}
        if k not in self.rows:
            return {"error": "unknown id"}
        r = self.rows[k]
        is_best = _num(str(value)) is not None and _num(str(value)) == _num(str(r["best"]))
        if tool == "checker":
            p = self._p(r, value)
            if p is None:
                return {"result": "no checker evidence for this value (it is not one of the candidates)"}
            agree = bool(r["consistent"]) if is_best else None
            return {"p_written_is_value": round(p, 3), "p_written_is_current": round(1 - p, 3),
                    "both_orders_agree": agree if agree is not None else "not recorded"}
        if tool == "beams":
            b = [c for c in str(r.get("beams", "") if pd.notna(r.get("beams", "")) else "").split("|") if c]
            ps = self.beam_p.get(int(r["i"]), [])
            return {"reader_top_numbers": [{"value": c, "p": round(ps[j], 3) if j < len(ps) else None}
                                           for j, c in enumerate(b)]}
        if tool == "page_count":
            v = _num(str(value))
            if v is None:
                return {"error": "value is not a number"}
            return {"times_elsewhere_on_page": sum(1 for j, x in self.nums if j != int(r["i"]) and x == v)}
        if tool == "blank":
            pb = r.get("p_blank")
            if not is_best or pd.isna(pb):
                return {"result": "not measured for this value"}
            return {"p_value_without_page": round(float(pb), 3), "p_value_with_page": round(float(r["p_change"]), 3)}
        if tool == "answer_key":
            v = _num(str(value))
            return {"in_answer_key": bool(v is not None and v in self.key)}
        return {"error": f"unknown tool {tool}"}


def advocate_prompt(entries: list[dict], results: list | None) -> str:
    led = "\n".join(f"ID {e['id']}: line: {e['line']!r} | written as {e['written']} | candidates {e['candidates']}"
                    for e in entries)
    head = f"{ROLE['advocate']}\n{RULES}\n\nFLAGGED NUMBERS (the [?] in each line is the number):\n{led}\n\n"
    if results is None:
        return head + (f"ROUND 1: call up to {TOOL_BUDGET} tools to examine the most suspicious IDs. Tools: "
                       "checker(id, value) = probability the student wrote value (from the image); beams(id) = the "
                       "reader's own top numbers at that spot; page_count(id, value) = how often value appears "
                       'elsewhere on the page. Reply: {"tool_calls": [{"tool": "checker", "id": 0, "value": "54"}, ...]}')
    return head + ("TOOL RESULTS:\n" + "\n".join(json.dumps(x) for x in results) +
                   '\n\nROUND 2: file claims only where the evidence says the student wrote a different value. Reply: '
                   '{"claims": [{"id": 0, "change_to": "54", "evidence": "checker 0.97, appears twice on the page"}]} '
                   '(an empty list is fine).')


def auditor_prompt(entries: list[dict], claims: list[dict], results: list | None) -> str:
    by = {e["id"]: e for e in entries}
    cl = "\n".join(f"CLAIM {j}: ID {c['id']} ({by[c['id']]['line']!r}, written as {by[c['id']]['written']}) -> "
                   f"{c['change_to']}; Advocate's evidence: {c.get('evidence', '')}" for j, c in enumerate(claims))
    head = f"{ROLE['auditor']}\n{RULES}\n\nTHE ADVOCATE'S CLAIMS:\n{cl}\n\n"
    if results is None:
        return head + (f"ROUND 1: call up to {TOOL_BUDGET} tools. Tools: checker(id, value); page_count(id, value); "
                       "blank(id, value) = the checker's preference for value WITHOUT the page image (high = guessed from "
                       "context); answer_key(id, value) = is value the expected answer in the rubric. Reply: "
                       '{"tool_calls": [{"tool": "blank", "id": 0, "value": "54"}, ...]}')
    return head + ("TOOL RESULTS:\n" + "\n".join(json.dumps(x) for x in results) +
                   '\n\nROUND 2: a verdict for every claim. Reply: {"verdicts": [{"claim": 0, "verdict": "support", '
                   '"reason": "..."}]} with verdict "support" or "object".')


def run_tools(ev: Evidence, calls, allowed: tuple, stats: dict) -> list[dict]:
    out = []
    for c in (calls or [])[:TOOL_BUDGET]:
        if not isinstance(c, dict) or c.get("tool") not in allowed:
            stats["invalid_tool_calls"] = stats.get("invalid_tool_calls", 0) + 1
            continue
        res = ev.call(c["tool"], c.get("id"), c.get("value"))
        stats[f"tool_{c['tool']}"] = stats.get(f"tool_{c['tool']}", 0) + 1
        if "error" in res:
            stats["tool_errors"] = stats.get("tool_errors", 0) + 1
        out.append({"call": {k: c.get(k) for k in ("tool", "id", "value")}, "result": res})
    return out


def run_page(ev: Evidence, llm) -> dict:
    """The four agent turns for one page. Returns claims (with the Auditor's verdicts) and the dialogue."""
    entries = [ev.entry(k) for k in ev.rows]
    stats: dict = {}
    r1 = parse_json(llm(advocate_prompt(entries, None))) or {}
    res1 = run_tools(ev, r1.get("tool_calls"), ADVOCATE_TOOLS, stats)
    r2 = parse_json(llm(advocate_prompt(entries, res1))) or {}
    claims, seen = [], set()
    for c in r2.get("claims") or []:
        try:
            k = int(c.get("id"))
        except (TypeError, ValueError, AttributeError):
            stats["invalid_claims"] = stats.get("invalid_claims", 0) + 1
            continue
        v = str(c.get("change_to", "")).strip()
        if k not in ev.rows or _num(v) is None or k in seen or _num(v) == _num(ev.entry(k)["written"]):
            stats["invalid_claims"] = stats.get("invalid_claims", 0) + 1
            continue
        seen.add(k)
        claims.append({"id": k, "change_to": v, "evidence": str(c.get("evidence", ""))[:300]})
    verdicts = {}
    if claims:
        a1 = parse_json(llm(auditor_prompt(entries, claims, None))) or {}
        res2 = run_tools(ev, a1.get("tool_calls"), AUDITOR_TOOLS, stats)
        a2 = parse_json(llm(auditor_prompt(entries, claims, res2))) or {}
        for v in a2.get("verdicts") or []:
            try:
                verdicts[int(v.get("claim"))] = str(v.get("verdict", "")).lower()
            except (TypeError, ValueError, AttributeError):
                pass
    for j, c in enumerate(claims):
        c["auditor"] = "object" if verdicts.get(j) == "object" else "support" if verdicts.get(j) == "support" else "none"
    return {"claims": claims, "stats": stats, "advocate_tools": res1}


def decide(df: pd.DataFrame, claim_of: dict, folds: np.ndarray, max_fc: float) -> dict[str, np.ndarray]:
    """Actions per variant (rows aligned with df). claim_of: row index -> claim dict."""
    from consistency import pick_tau_allowed
    from correct import CHANGE, actions, pick_tau
    n = len(df)
    claimed = np.array([k in claim_of for k in range(n)])
    # the claim must be for the value the checker evidence is about (the best challenger)
    same = np.array([k in claim_of and _num(claim_of[k]["change_to"]) == _num(str(df.iloc[k]["best"])) for k in range(n)])
    not_obj = np.array([k in claim_of and claim_of[k]["auditor"] != "object" for k in range(n)])
    approved, adv = claimed & same & not_obj, claimed & same
    out = {v: np.full(n, "keep", dtype=object) for v in ("pipeline", "agents", "agents_filter", "advocate_only")}
    for g in np.unique(folds):
        te, tr = folds == g, folds != g
        t_pipe = pick_tau(df[tr], 0.35, 0.15, max_fc)
        base = actions(df[te], t_pipe, 0.35, 0.15) == CHANGE
        out["pipeline"][te] = np.where(base, CHANGE, "keep")
        out["agents_filter"][te] = np.where(base & approved[te], CHANGE, "keep")
        for name, mask in (("agents", approved), ("advocate_only", adv)):
            t = pick_tau_allowed(df[tr], mask[tr], 0.35, 0.15, max_fc)
            ch = (actions(df[te], t, 0.35, 0.15) == CHANGE) & mask[te]
            out[name][te] = np.where(ch, CHANGE, "keep")
    return out


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="a correction run with verify*.csv (+ beams*.csv, labels.csv)")
    p.add_argument("--target-root", required=True)
    p.add_argument("--model-dir", default="qwen3b_ft")
    p.add_argument("--edu-data", default=os.environ.get("EDU_DATA"))
    p.add_argument("--llm", default="hf-text:Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--workers", type=int, default=16, help="pages in flight (the 7B batches across them)")
    p.add_argument("--max-false-change", type=float, default=0.01)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--limit", type=int, default=None, help="pages (quick test)")
    a = p.parse_args()
    from consistency import load_pages
    from correct import summarise, write_variants
    from edu_silent import key_sets
    from llm import Cache, call_llm
    from selector import student_folds

    d = Path(a.dir)
    out = d / "agents"
    out.mkdir(exist_ok=True)
    df = pd.concat([pd.read_csv(x, dtype={"gold": str, "best": str, "text": str, "cands": str, "beams": str,
                                          "p_cands": str}) for x in sorted(d.glob("verify*.csv"))], ignore_index=True)
    labels = pd.read_csv(d / "labels.csv", dtype={"gold": str, "gold_raw": str})
    pages = load_pages(df, Path(a.target_root), a.model_dir)
    keys = key_sets(df.assign(page=df["page"]), Path(a.edu_data)) if a.edu_data else {}
    beam_p: dict = {}
    for f in sorted(d.glob("beams*.csv")):
        for _, b in pd.read_csv(f, dtype={"beam_p": str}).iterrows():
            beam_p[(b["page"], int(b["i"]))] = [float(x) for x in str(b["beam_p"]).split("|") if x not in ("", "nan")]
    page_ids = [pg for pg in df["page"].unique() if pg in pages][: a.limit]
    cache = Cache(out / "agents_llm_cache.sqlite")
    llm = (lambda prompt: call_llm(prompt, a.llm, temperature=0.0, cache=cache))
    ledger_path = out / "ledger.jsonl"
    done = {}
    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["page"]] = r

    def one(pg):
        rows = df[df["page"] == pg]
        q = re.sub(r"_[1-9]$", "", pg.split("|")[2])
        ev = Evidence(rows, pages[pg], keys.get(q, set()),
                      {i: beam_p.get((pg, i), []) for i in rows["i"].astype(int)})
        res = run_page(ev, llm)
        return {"page": pg, "row_index": [int(x) for x in rows.index], **res}

    todo = [pg for pg in page_ids if pg not in done]
    print(f"{len(todo)} pages for the agents ({len(done)} done), LLM {a.llm}")
    from tqdm import tqdm
    with open(ledger_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        for rec in tqdm(ex.map(one, todo), total=len(todo), desc="agents"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[rec["page"]] = rec
    # claims -> rows of df
    claim_of, stats = {}, {}
    for rec in done.values():
        for k, v in rec["stats"].items():
            stats[k] = stats.get(k, 0) + v
        for c in rec["claims"]:
            claim_of[rec["row_index"][c["id"]]] = c
    sub = df[df["page"].isin(list(done))].reset_index(drop=False)
    claim_sub = {j: claim_of[orig] for j, orig in enumerate(sub["index"]) if orig in claim_of}
    folds = student_folds(sub["student"], min(a.folds, sub["student"].nunique()))
    acts = decide(sub, claim_sub, folds, a.max_false_change)
    nums = labels[(labels["type"] == "num") & (labels["op"] != "ins") & labels["page"].isin(sub["page"])]
    before = int(nums["misread"].sum())
    n_claims = len(claim_sub)
    res = {"pages": len(done), "claims": n_claims,
           "claims_objected_by_auditor": sum(c["auditor"] == "object" for c in claim_sub.values()),
           "claims_for_the_checked_value": int(sum(_num(c["change_to"]) == _num(str(sub.iloc[j]["best"]))
                                                   for j, c in claim_sub.items())),
           "tool_and_format_stats": stats}
    from correct import fixed_mask
    fx = fixed_mask(sub)
    res["claims_that_would_fix"] = int(sum(fx[j] and _num(c["change_to"]) == _num(str(sub.iloc[j]["best"]))
                                           for j, c in claim_sub.items()))
    for name, act in acts.items():
        s = summarise(sub, act, before, sub["page"].nunique())
        s["net_fixes"] = s["auto_fixed"] - s["auto_wrong_fix"] - s["auto_broken_correct"]
        res[name] = {k: v for k, v in s.items() if not k.startswith("ta_") and "perfect_ta" not in k}
    (out / "agents_metrics.json").write_text(json.dumps(res, indent=2))
    write_variants(out, sub, acts["agents"], labels, Path(a.target_root), a.model_dir)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
