"""Tests for the A/B checker code (no model, no data): python src/finetune/test_checker.py"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "detect"))
from ab_data import digit_edit, edu_pairs, fermat_pairs, lookalike, same_value  # noqa: E402
from checker_table import accuracy_table, best_checker  # noqa: E402
from choice_prompt import QUESTION, encode, masked_line  # noqa: E402
from eval_checker import pair_groups, sample, summarize  # noqa: E402
from train_checker import balance  # noqa: E402


def test_fermat_silent_pair_keeps_written_value():
    orig = r"x = 2 + 3 = 5 \Rightarrow y = 12 / 5"
    pert = r"x = 2 + 3 = 6 \Rightarrow y = 12 / 6"
    pairs = fermat_pairs(7, pert, orig, "train", random.Random(0))
    silent = [q for q in pairs if q["kind"] == "silent"]
    assert len(silent) == 4, silent  # two changed numbers x two orders
    for q in silent:
        right = q["a"] if q["answer"] == "A" else q["b"]
        wrong = q["b"] if q["answer"] == "A" else q["a"]
        assert right == "6" and wrong == "5" and q["written"] == "6"
        assert "[?]" in q["line"] and q["fermat_idx"] == 7
    assert {q["answer"] for q in silent} == {"A", "B"}
    look = [q for q in pairs if q["kind"] == "lookalike"]
    assert look and all((q["a"] if q["answer"] == "A" else q["b"]) == q["written"] for q in look)


def test_fermat_clean_row_has_only_lookalikes():
    s = "a = 4 + 4 = 8"
    pairs = fermat_pairs(0, s, s, "test", random.Random(0))
    assert pairs and all(q["kind"] == "lookalike" for q in pairs)
    assert len(pairs) == 2 * 2  # LOOKALIKE_PER_ROW pairs, both orders


def test_lookalike_and_digit_edit():
    rng = random.Random(1)
    for x in ["38", "1.5", "7", "100"]:
        c = lookalike(x, rng)
        assert c is not None and not same_value(c, x) and digit_edit(c, x) == 1
    assert digit_edit("59", "54") == 1 and digit_edit("59", "5") == 1 and digit_edit("59", "123") == 3
    assert same_value("0.228", ".228") and same_value("72,000", "72000")


def test_edu_pairs_use_near_misses_only():
    answer = "We get $V = 59 V$ and $I = 123 A$ then $R = 7$"
    rows = [  # transcript says 59 (expert 54: near miss), 123 (expert 9: far, alignment noise), 7 (correct)
        {"start": answer.index("59"), "end": answer.index("59") + 2, "raw": "59", "misread": 1, "gold_raw": "54",
         "student": "student_1", "page": "hw|student_1|p1"},
        {"start": answer.index("123"), "end": answer.index("123") + 3, "raw": "123", "misread": 1, "gold_raw": "9",
         "student": "student_1", "page": "hw|student_1|p1"},
        {"start": answer.index("7"), "end": answer.index("7") + 1, "raw": "7", "misread": 0, "gold_raw": "",
         "student": "student_1", "page": "hw|student_1|p1"},
    ]
    pairs = edu_pairs(rows, answer, "img.png", "test", random.Random(0))
    mis = [q for q in pairs if q["kind"] == "misread"]
    assert len(mis) == 2 and all(q["written"] == "54" for q in mis)
    assert all(q["line"].count("[?]") == 1 and "59" not in q["line"] for q in mis)
    look = [q for q in pairs if q["kind"] == "lookalike"]
    assert len(look) == 2 and look[0]["written"] == "7"


def test_masked_line_and_question():
    ans = "first line\nV = 59 + 3\nlast"
    s = ans.index("59")
    assert masked_line(ans, s, s + 2) == "V = [?] + 3"
    q = QUESTION.format(line="V = [?]", a="59", b="54")
    assert "A) 59" in q and "B) 54" in q

    class FakeProc:  # encode() must put the image and the question in one user turn, ending at the answer
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
            assert add_generation_prompt and msgs[0]["content"][0]["type"] == "image"
            return "PROMPT:" + msgs[0]["content"][1]["text"]

        def __call__(self, text, images, return_tensors):
            return {"text": text, "images": images}
    out = encode(FakeProc(), "IMG", "V = [?]", "59", "54")
    assert out["images"] == ["IMG"] and "A) 59" in out["text"][0]


def _pairs(kind, source, n):
    out = []
    for i in range(n):
        base = {"source": source, "kind": kind, "line": f"l{i}", "written": "1", "fermat_idx": i, "page": None}
        out += [{**base, "a": "1", "b": "7", "answer": "A"}, {**base, "a": "7", "b": "1", "answer": "B"}]
    return out


def test_balance_shares_budget_and_keeps_orders_together():
    pairs = _pairs("lookalike", "fermat", 500) + _pairs("silent", "fermat", 300) + _pairs("misread", "edu", 20)
    out = balance(pairs, max_examples=400)
    assert len(out) == 400
    kinds = {}
    for q in out:
        kinds[q["kind"]] = kinds.get(q["kind"], 0) + 1
    assert kinds["misread"] == 40  # all 20 scarce pairs kept (x2 orders)
    assert kinds["lookalike"] == 180 and kinds["silent"] == 180  # the rest shared evenly
    groups = pair_groups(out)
    assert len(groups) == 200 and all({q["answer"] for q in g} == {"A", "B"} for g in groups)


def test_eval_summary_and_best():
    groups = pair_groups(_pairs("misread", "edu", 10))
    assert len(sample(groups, 4)) == 4 and sample(groups, 4) == sample(groups, 4)
    rows = [{"group": "edu/misread", "p_right": p, "both_right": p > 0.6} for p in (0.9, 0.7, 0.4, 0.2)]
    s = summarize(rows)["edu/misread"]
    assert s == {"pairs": 4, "accuracy": 0.5, "both_orders": 0.5, "mean_p": 0.55}
    summary = {"reader": {"edu/misread": {"accuracy": 0.45, "both_orders": 0.3, "pairs": 9}},
               "checker_f": {"edu/misread": {"accuracy": 0.6, "both_orders": 0.5, "pairs": 9}},
               "checker_fe": {"edu/misread": {"accuracy": 0.7, "both_orders": 0.6, "pairs": 9}}}
    assert best_checker(summary) == "checker_fe"
    table = accuracy_table(summary)
    assert "| reader | checker_f | checker_fe |" in table and "70.0%" in table


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"all {len(tests)} checker tests passed")
