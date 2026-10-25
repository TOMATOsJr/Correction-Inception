"""Tests for the deterministic core (run: python src/agents/test_core.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import ACCEPT, ESCALATE, REJECT, Claim, apply_accepted, arbitrate, consistency_check, span_check  # noqa: E402

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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
