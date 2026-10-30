"""Tests for the deterministic core (run: python src/agents/test_core.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (ACCEPT, ESCALATE, REJECT, ArbiterConfig, Claim, apply_accepted, arbitrate,  # noqa: E402
                  consistency_check, span_check)
from protocol import parse_actions, parse_tool_requests  # noqa: E402

# Student wrote R = 2+3 = 6 (slip) and carried 6 forward; the transcriber silently corrected it to 5.
T = "R = 2 + 3 = 5\nI = 12 / 6 = 2\nP = 4 * 6 = 24"
# Student wrote V = 3*4 = 12 correctly; the transcriber misread 3 as 8.
T2 = "V = 8 * 4 = 12\nP = 2 * 12 = 24"


def test_span_check():
    assert span_check(T, "2 + 3 = 5") and not span_check(T, "2 + 3 = 7")


def test_consistency_never_uses_correctness():
    # proposing '6' for the result 5: later lines use 6 as an operand...
    r = consistency_check(T, "= 5", "= 6")
    assert r["support"] > 0 and r["reuse_signal"] == 1  # the student reused 6 later, never 5
    # the opposite proposal (student wrote 5) is NOT supported, even though 5 is the correct answer
    assert consistency_check("R = 2 + 3 = 6\nI = 12 / 6 = 2", "= 6", "= 5")["support"] < 0
    # misread operand: proposing 3 instead of 8 makes the student's OWN equality 3*4=12 hold
    r2 = consistency_check(T2, "8", "3")
    assert r2["support"] >= 0


def test_arbiter_accepts_supported_auditor_claim():
    c = Claim("auditor", "M", "2 + 3 = 5", "2 + 3 = 6")
    out = arbitrate([c], T, reread=lambda cl: ("2 + 3 = 6", 0.93))
    assert out[0].decision == ACCEPT, out[0].decision_reason
    assert "2 + 3 = 6" in apply_accepted(T, out)


def test_arbiter_rejects_when_reread_confirms_transcript():
    c = Claim("advocate", "E", "2 + 3 = 5", "2 + 3 = 8")
    assert arbitrate([c], T, reread=lambda cl: ("2 + 3 = 5", 0.95))[0].decision == REJECT


def test_conflict_and_diagram_escalate():
    a = Claim("advocate", "M", "12 / 6", "12 / 8")
    b = Claim("auditor", "M", "12 / 6", "12 / 5")
    d = Claim("auditor", "T", "series", "parallel", region_type="diagram")
    out = arbitrate([a, b, d], T, reread=lambda cl: ("x", 0.99))
    assert [c.decision for c in out] == [ESCALATE, ESCALATE, ESCALATE]


def test_invented_span_rejected():
    c = Claim("advocate", "M", "V = 99", "V = 98")
    assert arbitrate([c], T, reread=lambda cl: ("V = 98", 0.99))[0].decision == REJECT


def test_low_confidence_escalates():
    c = Claim("auditor", "M", "2 + 3 = 5", "2 + 3 = 6")
    assert arbitrate([c], T, reread=lambda cl: ("2 + 3 = 6", 0.4))[0].decision == ESCALATE



def test_reread_only_ignores_contradicting_later_work():
    # the re-read backs the claim but the student's later work contradicts it: full rules escalate,
    # re-read-only accepts (the consistency check is switched off)
    c = lambda: Claim("auditor", "M", "2 + 3 = 6", "2 + 3 = 5")  # noqa: E731
    t = "R = 2 + 3 = 6\nI = 12 / 6 = 2"
    assert arbitrate([c()], t, reread=lambda cl: ("2 + 3 = 5", 0.95))[0].decision == ESCALATE
    out = arbitrate([c()], t, reread=lambda cl: ("2 + 3 = 5", 0.95), cfg=ArbiterConfig(use_consistency=False))
    assert out[0].decision == ACCEPT


def test_consistency_only_ignores_the_reread():
    # later work supports the claim, the re-read says nothing: consistency-only accepts on later work alone
    c = lambda: Claim("auditor", "M", "2 + 3 = 5", "2 + 3 = 6")  # noqa: E731
    assert arbitrate([c()], T, reread=lambda cl: ("", 0.0))[0].decision == ESCALATE
    out = arbitrate([c()], T, reread=lambda cl: ("", 0.0), cfg=ArbiterConfig(use_reread=False))
    assert out[0].decision == ACCEPT and out[0].evidence["reread"] == ""
    # ... and rejects when the later work contradicts the claim
    bad = Claim("advocate", "M", "2 + 3 = 6", "2 + 3 = 5")
    assert arbitrate([bad], "R = 2 + 3 = 6\nI = 12 / 6 = 2", reread=lambda cl: ("2 + 3 = 5", 0.99),
                     cfg=ArbiterConfig(use_reread=False))[0].decision == REJECT

# ------------------------------------------------------------------ prompt parsing (protocol.py)
def test_parse_actions_single_backslash_latex():
    # invalid escapes (\O) used to fail the whole parse; valid ones (\f) silently mangled
    raw = '{"actions": [{"action": "new", "span": "\\frac{1}{48} + \\frac{1}{24} = 16 \\Omega", "proposed": "j50"}]}'
    acts = parse_actions(raw)
    assert len(acts) == 1
    assert acts[0]["span"] == "\\frac{1}{48} + \\frac{1}{24} = 16 \\Omega"


def test_parse_actions_valid_escape_mangle():
    # \frac inside a JSON string is a VALID escape (\f = formfeed): the old parser
    # "succeeded" and stored formfeed + "rac{...}", which then never matched the transcript
    raw = '{"claims": [{"rubric_code": "M", "span": "\\frac{1}{48}", "proposed": "\\frac{1}{24}"}]}'
    claims = parse_actions(raw)
    assert len(claims) == 1 and claims[0]["span"] == "\\frac{1}{48}"


def test_parse_actions_clean_json_untouched():
    # properly escaped LaTeX and legit unicode escapes must survive byte-for-byte
    raw = '{"actions": [{"action": "new", "span": "\\\\frac{1}{48} = 30\\u00b0", "proposed": "x"}]}'
    acts = parse_actions(raw)
    assert acts[0]["span"] == "\\frac{1}{48} = 30\u00b0"


def test_parse_actions_truncated_salvage():
    # responses are stored truncated; a balanced inner object must still be recovered
    raw = '{"actions": [{"action": "withdraw", "cid": "#1", "reason": "evidence does not support it"},'
    acts = parse_actions(raw)
    assert acts and acts[0]["action"] == "withdraw"


def test_parse_actions_malformed_salvage():
    # object closed early, stray key outside: the intact action must survive
    raw = '{"actions": [{"action": "counter", "cid": "#1", "span": "j50", "proposed": "j25"}, "for_claim": "#1"}]}'
    acts = parse_actions(raw)
    assert any(a.get("action") == "counter" for a in acts)


def test_parse_tool_requests_nested_in_actions():
    # some models put {"tool_requests": [...]} inside the actions list
    raw = '{"actions": [{"action": "new", "span": "a", "proposed": "b"}, ' \
          '{"tool_requests": [{"tool": "crop_reread", "args": {"span": "a", "proposed": "b"}}]}]}'
    assert len(parse_actions(raw)) == 1
    reqs = parse_tool_requests(raw)
    assert len(reqs) == 1 and reqs[0]["tool"] == "crop_reread"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
