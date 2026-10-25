"""Advocate / Auditor agents and the end-to-end correction pipeline.

  correct_transcript(transcript, rubric, reread_fn, model) ->
      {"claims": [...ledger...], "corrected": str, "escalated": [...], "accepted": [...]}

Agents see the transcript + rubric (not the grade) and file JSON claims; the deterministic Arbiter (core.py)
verifies each claim with the context-free re-read and the student's-own-work consistency check.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src" / "eval"))
from core import ACCEPT, ESCALATE, ArbiterConfig, Claim, apply_accepted, arbitrate, ledger_json  # noqa: E402
from llm import Cache, call_llm  # noqa: E402

ROLE = {
    "advocate": ("ADVOCATE", "You work for the STUDENT. Find places where the transcript is LIKELY A MISREAD of the "
                 "handwriting that would COST the student marks (e.g. 3 read as 8, a dropped minus sign, a missing "
                 "exponent, a wrong subscript) while the student's own surrounding work suggests they wrote something else."),
    "auditor": ("AUDITOR", "You protect GRADING INTEGRITY. Find places where the transcriber probably SILENTLY CORRECTED "
                "a student mistake, giving UNEARNED credit. Strong signal: a value/step that is suspiciously correct while "
                "the student's own LATER lines use a different (wrong) value, or a step that looks too clean compared with "
                "the rest of the work."),
}

PROMPT = """ROLE: {tag}
{mission}

Rules:
- Judge ONLY what the student wrote. Use the student's OWN later lines as evidence (values they reuse), NEVER whether the maths is correct.
- Only claim spans that change a rubric-relevant value or step. Quote the span EXACTLY as it appears in the transcript.
- One number/symbol change per claim. At most {max_claims} claims. If nothing is suspicious, return [].

RUBRIC CODES:
{codes}

TRANSCRIPT:
{transcript}

AUTOMATIC CHECKS (hints only, may be wrong):
{hints}

Return ONLY a JSON list: [{{"rubric_code": "...", "span": "<exact text>", "proposed": "<what the student actually wrote>",
"region_type": "text" | "diagram", "rationale": "<which later line(s) support this>"}}]"""


def _parse_claims(text: str, agent: str) -> list[Claim]:
    for cand in (text, *re.findall(r"```(?:json)?(.*?)```", text, re.S), *re.findall(r"\[.*\]", text, re.S)):
        try:
            data = json.loads(cand.strip())
            if isinstance(data, list):
                return [Claim(agent=agent, rubric_code=str(d.get("rubric_code", "")), span=str(d.get("span", "")),
                              proposed=str(d.get("proposed", "")), region_type=str(d.get("region_type", "text")),
                              rationale=str(d.get("rationale", ""))) for d in data if isinstance(d, dict)]
        except Exception:
            continue
    return []


def agent_prompt(agent: str, transcript: str, rubric: dict, hints: str = "", max_claims: int = 5) -> str:
    tag, mission = ROLE[agent]
    codes = "\n".join(f"- {c['code']}: {c.get('name', '')} - {c.get('description', '')}" for c in rubric.get("criteria", []))
    return PROMPT.format(tag=tag, mission=mission, codes=codes, transcript=transcript, max_claims=max_claims,
                         hints=hints or "(none)")


def run_agent(agent: str, transcript: str, rubric: dict, model: str, cache: Cache | None, max_claims: int = 5,
              hints: str = "") -> list[Claim]:
    out = call_llm(agent_prompt(agent, transcript, rubric, hints, max_claims), model, temperature=0.0, cache=cache)
    return [] if out.startswith("Error:") else _parse_claims(out, agent)


def correct_transcript(transcript: str, rubric: dict, reread_fn, model: str = "gemini-3.5-flash",
                       cache: Cache | None = None, agents=("advocate", "auditor"),
                       cfg: ArbiterConfig = ArbiterConfig()) -> dict:
    claims = [c for a in agents for c in run_agent(a, transcript, rubric, model, cache)]
    claims = arbitrate(claims, transcript, reread_fn, cfg)
    return {"claims": json.loads(ledger_json(claims)),
            "corrected": apply_accepted(transcript, claims),
            "accepted": [c.span + " -> " + c.proposed for c in claims if c.decision == ACCEPT],
            "escalated": [c.span for c in claims if c.decision == ESCALATE]}


if __name__ == "__main__":  # offline demo with scripted agent outputs
    import llm

    def fake(prompt, model, **kw):
        if "ROLE: AUDITOR" in prompt:
            return '[{"rubric_code":"M","span":"2 + 3 = 5","proposed":"2 + 3 = 6","region_type":"text","rationale":"line 2 uses 6"}]'
        return '[{"rubric_code":"E","span":"8 * 4","proposed":"3 * 4","region_type":"text","rationale":"3*4=12 matches"}]'

    globals()["call_llm"] = fake
    T = "R = 2 + 3 = 5\nI = 12 / 6 = 2\nV = 8 * 4 = 12\nP = 2 * 12 = 24"
    rub = {"criteria": [{"code": "M", "name": "Method"}, {"code": "E", "name": "Equation"}]}
    rereads = {"2 + 3 = 5": ("2 + 3 = 6", 0.92), "8 * 4": ("3 * 4", 0.90)}
    res = correct_transcript(T, rub, reread_fn=lambda c: rereads.get(c.span, ("", 0.0)), model="mock")
    print("accepted:", res["accepted"])
    print("escalated:", res["escalated"])
    print("corrected transcript:\n" + res["corrected"])
    assert res["corrected"] == "R = 2 + 3 = 6\nI = 12 / 6 = 2\nV = 3 * 4 = 12\nP = 2 * 12 = 24"
    print("DEMO PASSED")
