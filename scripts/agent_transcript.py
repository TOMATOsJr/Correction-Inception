"""Render the agent negotiation (r1 -> tools -> r2 -> tools -> r3 -> decide) as a
human-readable markdown transcript, one file per exam.

  python scripts/agent_transcript.py --out results/agents_smoke --name smoke3
  python scripts/agent_transcript.py --out results/agents_smoke --name smoke3 --exam "Homework1|student_31|3_6-1"

Reads only the JSONLs the run already wrote (r1/tools/r2/r3, interactive_ledger,
ledger) — no re-computation, safe to run any time, including mid-run.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

STATUS_ICON = {"open": "filed", "final": "STOOD", "withdrawn": "WITHDRAWN",
               "conceded": "CONCEDED", "escalate": "ESCALATED"}


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def claim_line(c: dict) -> str:
    return (f"`{c.get('span', '?')}` → `{c.get('proposed', '?')}` "
            f"*(rubric {c.get('rubric_code', '?')}, {c.get('region_type', '?')})*  \n{c.get('rationale', '')}")


def tool_lines(recs: list[dict], rnd: int, exam: str) -> list[str]:
    out = []
    for r in recs:
        if r.get("round") != rnd or r.get("exam") != exam:
            continue
        args = ", ".join(f"{k}=`{v}`" for k, v in (r.get("args") or {}).items())
        tool = r.get("tool", "?")
        if r.get("ok"):
            extra = {k: v for k, v in r.items() if k not in ("id", "exam", "agent", "round", "cid", "tool", "args", "ok")}
            detail = f" → re-read: `{extra.get('reading', '')}`" if extra.get("reading") else ""
            out.append(f"- **{r.get('agent', '?')}** called `{tool}`({args}){detail}")
        else:
            out.append(f"- **{r.get('agent', '?')}** called `{tool}`({args}) — ❌ rejected: {r.get('error', '?')}")
    return out


def action_lines(recs: list[dict], rnd: int) -> list[str]:
    out = []
    for r in recs:
        if r.get("round") != rnd:
            continue
        who = r["agent"]
        for a in r.get("actions") or []:
            if a.get("action") in ("defend", "stand", "withdraw", "concede"):
                out.append(f"- **{who}** {a['action']}s `{a.get('cid', '?')}`"
                           + (f": *{a.get('reason', '')}*" if a.get("reason") else ""))
            else:
                out.append(f"- **{who}** files `{a.get('cid', '?')}`: `{a.get('span', a)}` → `{a.get('proposed', '')}`")
        for tr in r.get("tool_requests") or []:
            out.append(f"- **{who}** requests `{tr.get('tool')}`({tr.get('args')})")
    return out


def render_exam(exam: str, r1, tools, r2, r3, ileg, final_rows) -> str:
    md = [f"# Negotiation transcript — `{exam}`", ""]
    for rnd, recs, title in ((1, r1, "Round 1 — initial claims"),
                             (2, r2, "Round 2 — cross-examination (both see the full ledger)"),
                             (3, r3, "Round 3 — rebuttal on remaining conflicts")):
        md += [f"## {title}", ""]
        said = False
        for r in recs:
            if r.get("exam") != exam:
                continue
            if rnd == 1:
                for c in r.get("claims") or []:
                    md += [f"**{r['agent']}** files {c.get('cid', 'a claim')}:  ", claim_line(c), ""]
                    said = True
            said |= bool(action_lines([r], rnd)) or bool(tool_lines([r], rnd, exam))
            md += action_lines([r], rnd)
        md += tool_lines(tools, rnd, exam)
        if not said:
            md.append("*(nothing filed — every proposed span was rejected at validation)*")
        md.append("")

    if ileg:
        md += ["## Final ledger (after the Arbiter)", ""]
        raw_claims = ileg.get("claims") or {}
        claim_items = (raw_claims.items() if isinstance(raw_claims, dict)
                       else enumerate(sorted(raw_claims, key=lambda c: c.get("cid", ""))))
        for cid, c in claim_items:
            cid = c.get("cid", cid) if isinstance(c, dict) else cid
            icon = STATUS_ICON.get(c.get("status"), c.get("status", "?"))
            md.append(f"- `{cid}` **{c.get('agent')}** {icon}: `{c.get('span')}` → `{c.get('proposed')}`")
            for h in c.get("history") or []:
                if h.get("action"):
                    md.append(f"  - r{h.get('round')}: {h.get('action')}"
                              + (f" — *{h.get('reason', '')}*" if h.get("reason") else ""))
            if c.get("evidence"):
                for k, v in c["evidence"].items():
                    md.append(f"  - evidence `{k}`: {v}")
        for rej in ileg.get("rejected") or []:
            md.append(f"- rejected action: {rej}")
        md.append("")
    if final_rows:
        md += ["## Arbiter decisions", ""]
        for r in final_rows:
            md.append(f"- `{r.get('cid', '?')}` **{r.get('agent')}** → **{str(r.get('decision')).upper()}** "
                      f"(`{r.get('span')}` → `{r.get('proposed')}`)")
        md.append("")
    return "\n".join(md)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="run output dir, e.g. results/agents_smoke")
    p.add_argument("--name", required=True, help="run name, e.g. smoke3")
    p.add_argument("--exam", default=None, help="only this exam (key_of format: hw|student|qid)")
    a = p.parse_args()

    run = Path(a.out) / a.name
    r1, r2, r3 = load(run / "r1.jsonl"), load(run / "r2.jsonl"), load(run / "r3.jsonl")
    tools = load(run / "tools.jsonl")
    ilegs = {r["exam"]: r for r in load(run / "interactive_ledger.jsonl")}
    finals = defaultdict(list)
    for r in load(run / "ledger.jsonl"):
        finals[r.get("id")].append(r)

    exams = sorted({r.get("exam") for r in r1 + r2 + r3 + tools} | set(ilegs))
    if a.exam:
        exams = [e for e in exams if e == a.exam]
    dest = run / "transcripts"
    dest.mkdir(exist_ok=True)
    for exam in exams:
        text = render_exam(exam, r1, tools, r2, r3, ilegs.get(exam), finals.get(exam, []))
        f = dest / f"{exam.replace('|', '_')}.md"
        f.write_text(text, encoding="utf-8")
        print(f"wrote {f}")
    print(f"{len(exams)} transcript(s) in {dest}")


if __name__ == "__main__":
    main()
