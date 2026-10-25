"""Correction Inception: claim ledger, verification tools and the rule-based arbiter (deterministic core).

Agents (Advocate = misreads that COST marks, Auditor = silent corrections that GAVE marks) only file Claims.
Tools gather evidence. The Arbiter is fixed rules (not an LLM), so it cannot be persuaded:
  ACCEPT    re-read supports the claim with confidence >= theta  AND  consistency evidence does not contradict it
  ESCALATE  claims conflict, OR the evidence is weak/unverifiable for a mark-relevant claim, OR the region is a diagram
  REJECT    the evidence contradicts the claim
Rule: consistency = agreement with the student's OWN later work, never mathematical correctness.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Callable

ACCEPT, ESCALATE, REJECT = "accept", "escalate", "reject"


@dataclass
class Claim:
    agent: str                 # "advocate" | "auditor"
    rubric_code: str           # which mark it affects
    span: str                  # exact text in the transcript claimed to be wrong
    proposed: str              # what the agent believes the student actually wrote
    bbox: list | None = None   # image region [x0, y0, x1, y1] (normalised), if available
    region_type: str = "text"  # "text" | "diagram"
    rationale: str = ""
    evidence: dict = field(default_factory=dict)
    decision: str | None = None
    decision_reason: str = ""


# ------------------------------------------------------------------ tools
def span_check(transcript: str, span: str) -> bool:
    """The claimed span must really exist in the transcript (agents cannot invent text)."""
    norm = lambda s: re.sub(r"\s+", "", s)  # noqa: E731
    return bool(span) and norm(span) in norm(transcript)


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _safe_eval(expr: str):
    expr = expr.replace("^", "**").replace("\\times", "*").replace("\\cdot", "*")
    expr = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", expr).replace("{", "(").replace("}", ")")
    if not re.fullmatch(r"[0-9.+\-*/() ]+", expr.replace("**", "*")):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))  # whitelisted: digits/operators only
    except Exception:
        return None


def changed_number(span: str, proposed: str):
    """The single number a claim changes: (old, new), or None if the claim does not change exactly one number.
    Lets claims quote context ("8 * 4" -> "3 * 4") while still naming one disputed value."""
    a, b = _NUM.findall(span), _NUM.findall(proposed)
    if len(a) == 1 and len(b) == 1:
        return (a[0], b[0]) if a[0] != b[0] else None
    if len(a) == len(b):
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        return diff[0] if len(diff) == 1 else None
    return None


def consistency_check(transcript: str, span: str, proposed: str, rel_tol: float = 0.02) -> dict:
    """Does the PROPOSED reading fit the student's own later work better than the transcribed span?

    Takes every later numeric equality 'lhs = rhs' in which the span's number occurs as an operand, substitutes the
    proposed value, and counts which version makes the student's OWN later equalities hold. Never checks whether the
    student's answer is correct.
    Returns {"support": +k (proposed fits better) / -k (span fits better) / 0, "n_constraints": n}.
    """
    ch = changed_number(span, proposed)
    if ch is None:
        return {"support": 0, "n_constraints": 0, "note": "claim must change exactly one number"}
    old, new = ch
    pos = transcript.find(span)
    later = transcript[pos + len(span):] if pos >= 0 else transcript
    support, n = 0, 0
    for line in later.splitlines():
        parts = re.split(r"(?<![<>!])=(?!=)", line.strip("$ "))
        for lhs, rhs in zip(parts, parts[1:]):
            if not re.search(rf"(?<![\d.]){re.escape(old)}(?![\d.])", lhs):
                continue
            rv = _safe_eval(rhs)
            lv_old = _safe_eval(lhs)
            lv_new = _safe_eval(re.sub(rf"(?<![\d.]){re.escape(old)}(?![\d.])", new, lhs))
            if None in (rv, lv_old, lv_new):
                continue
            n += 1
            ok_old = abs(lv_old - rv) <= rel_tol * max(1e-9, abs(rv))
            ok_new = abs(lv_new - rv) <= rel_tol * max(1e-9, abs(rv))
            support += int(ok_new and not ok_old) - int(ok_old and not ok_new)
    # copy evidence: students reuse values they wrote. If the proposed value is reused later and the transcribed
    # one never is, the proposal is supported (and vice versa).
    occurs = lambda v: bool(re.search(rf"(?<![\d.]){re.escape(v)}(?![\d.])", later))  # noqa: E731
    reuse = int(occurs(new) and not occurs(old)) - int(occurs(old) and not occurs(new))
    if reuse:
        n += 1
        support += reuse
    return {"support": support, "n_constraints": n, "reuse_signal": reuse}


def retrieve_rubric(rubric: dict, code: str) -> dict | None:
    for c in rubric.get("criteria", []):
        if c.get("code") == code:
            return c
    return None


# ------------------------------------------------------------------ arbiter
@dataclass
class ArbiterConfig:
    theta: float = 0.85          # minimum re-read confidence to accept
    require_consistency_when_available: bool = True


def arbitrate(claims: list[Claim], transcript: str,
              reread: Callable[[Claim], tuple[str, float]], cfg: ArbiterConfig = ArbiterConfig()) -> list[Claim]:
    """Gather evidence for every claim and decide. `reread(claim) -> (reading, confidence)` wraps the
    faithful reader on the claim's image region (context-free)."""
    # conflicts: two claims on the same span proposing different readings -> human
    by_span: dict[str, set] = {}
    for c in claims:
        by_span.setdefault(re.sub(r"\s+", "", c.span), set()).add(re.sub(r"\s+", "", c.proposed))
    for c in claims:
        key = re.sub(r"\s+", "", c.span)
        if c.region_type == "diagram":
            c.decision, c.decision_reason = ESCALATE, "diagram regions always go to a human"
            continue
        if not span_check(transcript, c.span):
            c.decision, c.decision_reason = REJECT, "span not found in transcript"
            continue
        if len(by_span[key]) > 1:
            c.decision, c.decision_reason = ESCALATE, "conflicting claims on the same span"
            continue
        reading, conf = reread(c)
        cons = consistency_check(transcript, c.span, c.proposed)
        c.evidence = {"reread": reading, "reread_conf": conf, **cons}
        matches = re.sub(r"\s+", "", reading) == re.sub(r"\s+", "", c.proposed)
        keeps_span = re.sub(r"\s+", "", reading) == key
        if matches and conf >= cfg.theta and cons["support"] >= 0:
            if cfg.require_consistency_when_available and cons["n_constraints"] > 0 and cons["support"] == 0:
                c.decision, c.decision_reason = ESCALATE, "re-read agrees but later work is ambiguous"
            else:
                c.decision, c.decision_reason = ACCEPT, "re-read supports claim; later work does not contradict"
        elif keeps_span and conf >= cfg.theta and cons["support"] <= 0:
            c.decision, c.decision_reason = REJECT, "re-read confirms the transcript"
        elif cons["support"] > 0 and not keeps_span:
            c.decision, c.decision_reason = ESCALATE, "later work supports claim but re-read is weak"
        else:
            c.decision, c.decision_reason = ESCALATE, "evidence weak or contradictory"
    return claims


def apply_accepted(transcript: str, claims: list[Claim]) -> str:
    seen = set()  # both agents may file the same fix: apply it once
    for c in claims:
        if c.decision == ACCEPT and (c.span, c.proposed) not in seen:
            seen.add((c.span, c.proposed))
            transcript = transcript.replace(c.span, c.proposed, 1)
    return transcript


def ledger_json(claims: list[Claim]) -> str:
    return json.dumps([asdict(c) for c in claims], indent=2, ensure_ascii=False)
