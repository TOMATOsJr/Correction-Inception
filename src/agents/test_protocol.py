"""Tests for the interactive agent protocol (run: python src/agents/test_protocol.py).

Covers the guarantees the protocol is supposed to provide:
  * state changes only through validated actions (ownership, round legality, span existence);
  * tool calls are sandboxed, budgeted and never raise;
  * the ledger replays deterministically from its JSONL records;
  * disagreement between the agents is detected and routed to round 3;
  * the arbiter accepts a claim only with evidence that came from the ledger.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import ACCEPT, ESCALATE, ArbiterConfig, arbitrate, conflicting_indices  # noqa: E402
from protocol import (Ledger, ToolBox, apply_actions, apply_new, build_ledger, conflict_groups,  # noqa: E402
                      parse_actions, parse_single_action, parse_tool_requests)

# the student wrote 3 * 4 = 12 and reused 12 later; the transcriber misread 3 as 8
T = "V = 8 * 4 = 12\nP = 2 * 12 = 24\nQ = 12 / 4 = 3"
RUBRIC = {"criteria": [{"code": "M", "name": "Method", "max_deduction": 1},
                       {"code": "E", "name": "Equation", "max_deduction": 1}]}


def _ledger():
    return Ledger(exam="Homework1|student_21|1_5-2", transcript=T)


# ------------------------------------------------------------------ round 1
def test_new_claim_requires_a_real_span():
    led = _ledger()
    rec, err = apply_new(led, "advocate", {"rubric_code": "M", "span": "8 * 4", "proposed": "3 * 4"}, 1)
    assert rec is not None and err is None and rec.status == "open"
    rec2, err2 = apply_new(led, "advocate", {"rubric_code": "M", "span": "V = 99", "proposed": "V = 98"}, 1)
    assert rec2 is None and "not found" in err2
    rec3, err3 = apply_new(led, "advocate", {"rubric_code": "M", "span": "8", "proposed": "8"}, 1)
    assert rec3 is None and err3


def test_action_must_be_legal_for_the_round():
    led = _ledger()
    apply_new(led, "advocate", {"span": "8 * 4", "proposed": "3 * 4"}, 1)
    assert apply_actions(led, "advocate", 1, [{"action": "concede", "cid": "#1"}])[0]["ok"] is False
    assert apply_actions(led, "advocate", 2, [{"action": "concede", "cid": "#1"}])[0]["ok"] is False
    assert apply_actions(led, "advocate", 3, [{"action": "concede", "cid": "#1"}])[0]["ok"] is True


def test_agent_cannot_touch_the_other_agents_claim():
    led = _ledger()
    apply_new(led, "auditor", {"span": "8 * 4", "proposed": "6 * 4"}, 1)
    out = apply_actions(led, "advocate", 2, [{"action": "withdraw", "cid": "#1"}])
    assert out[0]["ok"] is False and "your own" in out[0]["error"]
    assert led.claims["#1"].status == "open"


def test_revise_supersedes_and_counter_links():
    led = _ledger()
    apply_new(led, "advocate", {"span": "8 * 4", "proposed": "3 * 4"}, 1)
    apply_actions(led, "advocate", 2, [{"action": "revise", "cid": "#1", "span": "8", "proposed": "3"}])
    assert led.claims["#1"].status == "superseded" and led.claims["#1"].superseded_by == "#2"
    apply_new(led, "auditor", {"span": "8 * 4", "proposed": "6 * 4"}, 1)
    apply_actions(led, "advocate", 2, [{"action": "counter", "cid": "#3", "proposed": "3 * 4"}])
    assert led.claims["#4"].counters == "#3" and led.claims["#4"].span == "8 * 4"


def test_malformed_actions_never_crash():
    led = _ledger()
    apply_new(led, "advocate", {"span": "8 * 4", "proposed": "3 * 4"}, 1)
    out = apply_actions(led, "advocate", 2, ["nonsense", {"reason": "no action"}, {"action": "withdraw", "cid": "#99"}])
    assert [r["ok"] for r in out] == [False, False, False]
    assert len(led.rejected) == 3


# ------------------------------------------------------------------ conflicts / round 3
def test_conflict_detection_covers_overlapping_spans():
    led = _ledger()
    apply_new(led, "advocate", {"span": "8 * 4", "proposed": "3 * 4"}, 1)
    apply_new(led, "auditor", {"span": "8 * 4 = 12", "proposed": "6 * 4 = 12"}, 1)
    groups = conflict_groups(led)
    assert len(groups) == 1 and set(groups[0]) == {"#1", "#2"}
    # agreement is not a conflict
    led2 = _ledger()
    apply_new(led2, "advocate", {"span": "8 * 4", "proposed": "3 * 4"}, 1)
    apply_new(led2, "auditor", {"span": "8 * 4", "proposed": "3 * 4"}, 1)
    assert conflict_groups(led2) == []


def test_concede_clears_the_conflict():
    led = _ledger()
    apply_new(led, "advocate", {"span": "8 * 4", "proposed": "3 * 4"}, 1)
    apply_new(led, "auditor", {"span": "8 * 4", "proposed": "6 * 4"}, 1)
    apply_actions(led, "auditor", 3, [{"action": "concede", "cid": "#2", "reason": "re-read says 3"}])
    assert conflict_groups(led) == [] and led.claims["#2"].status == "conceded"
    assert len(led.final_claims()) == 1


# ------------------------------------------------------------------ tools
def test_toolbox_budget_unknown_tool_and_missing_reader():
    box = ToolBox(T, RUBRIC, reread=lambda span, proposed: ("3 * 4", 0.91), budget=1)
    assert box.call("advocate", "nope", {})["ok"] is False
    res = box.call("advocate", "consistency", {"span": "8 * 4", "proposed": "3 * 4"})
    assert res["ok"] and res["result"]["support"] >= -5   # consistency evidence, not a crash
    over = box.call("advocate", "span_check", {"span": "8 * 4"})   # budget already spent
    assert over["ok"] is False and "budget" in over["error"]
    no_reader = ToolBox(T, RUBRIC, reread=None, budget=4)
    assert no_reader.call("auditor", "crop_reread", {"span": "8 * 4", "proposed": "3 * 4"})["ok"] is False


def test_crop_reread_resolves_to_the_claim_reading():
    box = ToolBox(T, RUBRIC, reread=lambda span, proposed: ("3 * 4", 0.91), budget=2)
    res = box.call("advocate", "crop_reread", {"span": "8 * 4", "proposed": "3 * 4"})
    assert res["ok"] and res["result"]["resolved"] == "3 * 4" and res["result"]["conf"] == 0.91
    # a claim that changes two numbers cannot be re-read
    assert box.call("advocate", "crop_reread", {"span": "8 * 4", "proposed": "3 * 7"})["ok"] is False


# ------------------------------------------------------------------ replay + arbiter
def test_replay_is_deterministic():
    r1 = [{"exam": "e", "agent": "advocate", "round": 1,
           "claims": [{"span": "8 * 4", "proposed": "3 * 4", "rubric_code": "M"}]},
          {"exam": "e", "agent": "auditor", "round": 1,
           "claims": [{"span": "8 * 4 = 12", "proposed": "6 * 4 = 12", "rubric_code": "M"}]}]
    tools = [{"id": "t1", "exam": "e", "agent": "advocate", "round": 1, "cid": "#1",
              "tool": "crop_reread", "ok": True, "result": {"reading": "3 * 4", "conf": 0.91, "resolved": "3 * 4"}}]
    actions = [{"exam": "e", "agent": "auditor", "round": 3, "actions": [{"action": "concede", "cid": "#2"}]}]
    a = build_ledger("e", T, r1, tools, actions)
    b = build_ledger("e", T, r1, tools, actions)
    assert [(c.cid, c.agent, c.status) for c in a.claims.values()] == [(c.cid, c.agent, c.status) for c in b.claims.values()]
    assert a.claims["#1"].evidence["crop_reread"]["resolved"] == "3 * 4"
    assert a.claims["#2"].status == "conceded"


def test_arbiter_uses_ledger_evidence_and_escalates_conflicts():
    led = _ledger()
    apply_new(led, "advocate", {"span": "8 * 4", "proposed": "3 * 4"}, 1)
    led.attach_tool(led.claims["#1"], {"tool": "crop_reread", "ok": True,
                                       "result": {"reading": "3 * 4", "conf": 0.93, "resolved": "3 * 4"}}, "t1")
    led.attach_tool(led.claims["#1"], {"tool": "consistency", "ok": True,
                                       "result": {"support": 1, "n_constraints": 1, "reuse_signal": 1}}, "t2")
    out = arbitrate(led.final_claims(), T, reread=lambda c: ("", 0.0), cfg=ArbiterConfig(), use_existing_evidence=True)
    assert out[0].decision == ACCEPT, out[0].decision_reason
    # a second agent disputing the same value sends both to a human
    led2 = _ledger()
    apply_new(led2, "advocate", {"span": "8 * 4", "proposed": "3 * 4"}, 1)
    apply_new(led2, "auditor", {"span": "8 * 4 = 12", "proposed": "6 * 4 = 12"}, 1)
    out2 = arbitrate(led2.final_claims(), T, reread=lambda c: ("", 0.0), cfg=ArbiterConfig(), use_existing_evidence=True)
    assert [c.decision for c in out2] == [ESCALATE, ESCALATE]


def test_conflicting_indices_ignores_identical_proposals():
    led = _ledger()
    apply_new(led, "advocate", {"span": "8 * 4", "proposed": "3 * 4"}, 1)
    apply_new(led, "auditor", {"span": "8 * 4", "proposed": "3 * 4"}, 1)
    assert conflicting_indices(led.final_claims()) == set()


# ------------------------------------------------------------------ parsing
def test_parsers_tolerate_weak_model_output():
    assert parse_actions('```json\n{"actions": [{"action": "withdraw", "cid": "#1"}]}\n```')[0]["cid"] == "#1"
    assert parse_actions('[{"span": "a"}]')[0]["span"] == "a"
    assert parse_actions("Error: boom") == []
    assert parse_actions("sorry, no JSON here") == []
    assert parse_tool_requests('{"claims": [], "tool_requests": [{"tool": "rubric"}]}')[0]["tool"] == "rubric"
    assert parse_single_action('{"action": "stand", "cid": "#2"}')["action"] == "stand"
    assert parse_single_action("Error: x") is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
