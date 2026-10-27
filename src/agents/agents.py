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

TOOLS_DOC = """TOOLS (each call is executed for you and the result is shown to BOTH agents next round; {budget} per agent):
- `crop_reread` args {{"span": "<exact span>", "proposed": "<your reading>"}} -> the fine-tuned faithful reader
  re-reads that line ON THE PAGE IMAGE with the disputed number hidden, and reports what it saw + confidence.
  Use it when you are unsure what the student actually wrote.
- `consistency` args {{"span": "<exact span>", "proposed": "<your reading>"}} -> checks the student's OWN later
  lines: does your reading make their later equalities hold, and do they reuse your value? (never correctness)
- `span_check` args {{"span": "<exact span>"}} -> is this text really in the transcript?
- `rubric` args {{"code": "<code>"}} -> the full rubric criterion."""

PROMPT_R1 = """ROLE: {tag}
{mission}

This is ROUND 1: you file claims independently. The other agent is doing the same; you cannot see their
claims yet. Tool results you request now are shown to both agents in round 2.

Rules:
- Judge ONLY what the student wrote. Use the student's OWN later lines as evidence (values they reuse), NEVER whether the maths is correct.
- Only claim spans that change a rubric-relevant value or step. Quote the span EXACTLY as it appears in the transcript.
- One number/symbol change per claim. At most {max_claims} claims. If nothing is suspicious, file none.
- Request a tool when the evidence would change your mind, not to confirm what you already believe.

{tools}

RUBRIC CODES:
{codes}

TRANSCRIPT:
{transcript}

AUTOMATIC CHECKS (hints only, may be wrong):
{hints}

Return ONLY JSON:
{{"claims": [{{"rubric_code": "...", "span": "<exact text>", "proposed": "<what the student actually wrote>",
  "region_type": "text" | "diagram", "rationale": "<which later line(s) support this>"}}],
 "tool_requests": [{{"tool": "crop_reread", "args": {{"span": "...", "proposed": "..."}}, "for_claim": 0}}]}}
`for_claim` is the 0-based index of the claim the request is about (omit it if the request is not about a claim)."""

PROMPT_R2 = """ROLE: {tag}
{mission}

This is ROUND 2 (cross-examination). Below is the SHARED CLAIM LEDGER: both agents' claims and every tool
result gathered so far. Act on the ledger using these actions:

- `new`      file a claim you did not file before (e.g. because the other agent's evidence shows a spot they missed)
- `withdraw` drop one of YOUR claims that the evidence does not support
- `revise`   change one of YOUR claims to a better span or reading (creates a replacement claim)
- `counter`  file YOUR reading for a claim the OTHER agent filed (you think they misread the page)
- `defend`   keep YOUR claim and explain why the evidence against it is itself a misread

Rules:
- Judge ONLY what the student wrote, using their OWN later lines. NEVER whether the maths is correct.
- Withdraw rather than defend when the evidence (re-read or consistency) contradicts you. A clean ledger is worth
  more than a defended wrong claim.
- Do not duplicate a claim that already says the same thing; say nothing about claims you agree with.
- Request tools only when a result would change what you do next ({budget} per agent per exam).

{tools}

RUBRIC CODES:
{codes}

TRANSCRIPT:
{transcript}

AUTOMATIC CHECKS (hints only, may be wrong):
{hints}

SHARED CLAIM LEDGER:
{ledger}

Return ONLY JSON:
{{"actions": [{{"action": "withdraw|revise|counter|defend|new", "cid": "#2 (required for all but new)",
  "rubric_code": "...", "span": "<new/replacement span>", "proposed": "<new/replacement reading>",
  "reason": "<why>"}}],
 "tool_requests": [{{"tool": "crop_reread", "args": {{"span": "...", "proposed": "..."}}, "for_claim": "#2"}}]}}
Return {{"actions": []}} if nothing should change."""

PROMPT_R3 = """ROLE: {tag}
{mission}

This is ROUND 3 (rebuttal) on ONE disputed value. You and the other agent still propose DIFFERENT readings
for the same span. Choose exactly one:

- `stand`    you still believe your reading (a human reviews it if you both stand)
- `concede`  the other agent's reading is what the student wrote
- `withdraw` your claim was wrong and nothing should change in the transcript

Decide from the EVIDENCE, not from stubbornness: does the context-free re-read of the page agree with you
(and with what confidence), and does the student's own later work reuse your value?

{tools}

DISPUTED SPAN: `{span}`

CLAIMS DISPUTING IT:
{ledger}

TRANSCRIPT:
{transcript}

Return ONLY JSON: {{"action": "stand|concede|withdraw", "cid": "<your claim id>", "reason": "<short>",
 "tool_requests": [{{"tool": "crop_reread", "args": {{"span": "...", "proposed": "..."}}}}]}}"""


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


def _codes(rubric: dict) -> str:
    return "\n".join(f"- {c['code']}: {c.get('name', '')} - {c.get('description', '')}"
                     for c in rubric.get("criteria", []))


def agent_prompt(agent: str, transcript: str, rubric: dict, hints: str = "", max_claims: int = 5) -> str:
    """Round-1 prompt WITHOUT tools (the one-shot ablation path)."""
    tag, mission = ROLE[agent]
    return PROMPT.format(tag=tag, mission=mission, codes=_codes(rubric), transcript=transcript,
                         max_claims=max_claims, hints=hints or "(none)")


def agent_prompt_r1(agent: str, transcript: str, rubric: dict, hints: str = "", max_claims: int = 3,
                    tool_budget: int = 4) -> str:
    """Round 1 of the interactive protocol: claims + tool requests."""
    tag, mission = ROLE[agent]
    return PROMPT_R1.format(tag=tag, mission=mission, codes=_codes(rubric), transcript=transcript,
                            max_claims=max_claims, hints=hints or "(none)",
                            tools=TOOLS_DOC.format(budget=tool_budget), budget=tool_budget)


def agent_prompt_r2(agent: str, transcript: str, rubric: dict, ledger: str, hints: str = "",
                    tool_budget: int = 4) -> str:
    """Round 2: the shared ledger is in the prompt, so each agent can react to the other and to the evidence."""
    tag, mission = ROLE[agent]
    return PROMPT_R2.format(tag=tag, mission=mission, codes=_codes(rubric), transcript=transcript, hints=hints or "(none)",
                            ledger=ledger, tools=TOOLS_DOC.format(budget=tool_budget), budget=tool_budget)


def agent_prompt_r3(agent: str, transcript: str, span: str, ledger: str, tool_budget: int = 1) -> str:
    """Round 3: rebuttal on one disputed span."""
    tag, mission = ROLE[agent]
    return PROMPT_R3.format(tag=tag, mission=mission, span=span, ledger=ledger, transcript=transcript,
                            tools=TOOLS_DOC.format(budget=tool_budget), budget=tool_budget)


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
