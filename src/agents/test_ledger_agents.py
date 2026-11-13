"""Tests for the ledger agents (no model): python src/agents/test_ledger_agents.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger_agents import Evidence, decide, parse_json, run_page  # noqa: E402

PAGE = "$R = 54$ ohm, $V = 108$ so $I = 108 / 59 = 2$ and $P = 54 \\cdot 2 = 7$"


def rows():
    s59, s7 = PAGE.index("59"), PAGE.rindex("7")
    return pd.DataFrame([
        {"page": "p", "student": "s0", "i": 6, "start": s59, "end": s59 + 2, "text": "59", "gold": "54", "misread": 1,
         "cands": "54|58", "p_cands": "54:0.9700|58:0.1000", "best": "54", "p_change": 0.97, "consistent": True,
         "p_blank": 0.2, "beams": "54|58", "cand_is_key": False, "orig_is_key": False},
        {"page": "p", "student": "s0", "i": 11, "start": s7, "end": s7 + 1, "text": "7", "gold": "7", "misread": 0,
         "cands": "1|2", "p_cands": "1:0.6000|2:0.1000", "best": "1", "p_change": 0.6, "consistent": False,
         "p_blank": 0.55, "beams": "1", "cand_is_key": False, "orig_is_key": False}])


def test_parse_json():
    assert parse_json('blah {"a": {"b": 1}} tail') == {"a": {"b": 1}}
    assert parse_json("no json here") is None


def test_tools_answer_from_evidence():
    ev = Evidence(rows(), PAGE, {54.0}, {6: [0.6, 0.2]})
    assert ev.call("checker", 0, "54")["p_written_is_value"] == 0.97
    assert "no checker evidence" in ev.call("checker", 0, "99")["result"]
    assert ev.call("page_count", 0, "54")["times_elsewhere_on_page"] == 2
    assert ev.call("beams", 0)["reader_top_numbers"][0] == {"value": "54", "p": 0.6}
    assert ev.call("answer_key", 0, "54") == {"in_answer_key": True}
    assert ev.call("blank", 1, "1")["p_value_without_page"] == 0.55
    assert ev.call("checker", 9, "1") == {"error": "unknown id"}


def test_four_turns_and_verdicts():
    replies = iter([
        json.dumps({"tool_calls": [{"tool": "checker", "id": 0, "value": "54"}, {"tool": "page_count", "id": 0,
                                    "value": "54"}, {"tool": "blank", "id": 1, "value": "1"}]}),  # blank: not allowed
        json.dumps({"claims": [{"id": 0, "change_to": "54", "evidence": "checker .97, twice on page"},
                               {"id": 1, "change_to": "1"}, {"id": 1, "change_to": "2"}, {"id": 5, "change_to": "3"}]}),
        json.dumps({"tool_calls": [{"tool": "blank", "id": 1, "value": "1"}]}),
        json.dumps({"verdicts": [{"claim": 0, "verdict": "support"}, {"claim": 1, "verdict": "object"}]})])
    prompts = []

    def llm(p):
        prompts.append(p)
        return next(replies)
    res = run_page(Evidence(rows(), PAGE, set(), {}), llm)
    assert [(c["id"], c["change_to"], c["auditor"]) for c in res["claims"]] == [(0, "54", "support"), (1, "1", "object")]
    assert res["stats"]["invalid_tool_calls"] == 1 and res["stats"]["invalid_claims"] == 2
    assert "ROLE: ADVOCATE" in prompts[0] and "TOOL RESULTS" in prompts[1] and "ROLE: AUDITOR" in prompts[2]
    assert "[?]" in prompts[0] and "59" not in prompts[0].split("ID 0: line: ")[1].split("|")[0]


def test_arbiter_never_applies_an_objected_claim():
    n = 40
    df = pd.DataFrame([{"page": "p", "student": f"s{k % 4}", "i": k, "start": 0, "end": 1, "text": "59",
                        "gold": "54" if k % 2 == 0 else "59", "misread": int(k % 2 == 0), "best": "54",
                        "p_change": 0.99 if k % 2 == 0 else 0.4, "consistent": True, "p_blank": 0.1,
                        "cand_is_key": False, "orig_is_key": False} for k in range(n)])
    claims = {k: {"change_to": "54", "auditor": "object" if k % 4 == 0 else "support"} for k in range(0, n, 2)}
    acts = decide(df, claims, np.array([k % 4 for k in range(n)]), 0.05)
    assert all(acts["agents"][k] != "change" for k in claims if claims[k]["auditor"] == "object")
    assert any(acts["agents"][k] == "change" for k in claims if claims[k]["auditor"] == "support")
    assert sum(acts["advocate_only"] == "change") >= sum(acts["agents"] == "change")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL PASSED")
