"""Tests for misread labels and the detection metrics (run: python src/detect/test_detect.py). No model needed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_detection import at_k, auroc, average_precision  # noqa: E402
from labels import label_page, math_tokens  # noqa: E402
from score_tokens import aggregate  # noqa: E402


def test_tokens_keep_offsets_and_skip_formatting():
    text = r"so $\boxed{q = 72,000 \cdot t}$ and \[ \frac{1}{2} \tau \]"
    toks = math_tokens(text)
    assert [t["text"] for t in toks] == ["q", "=", "72000.0", "*", "t", "1.0", "2.0", "tau"]
    assert all(text[t["start"]:t["end"]] == t["raw"] for t in toks)  # offsets point into the original text


def test_numbers_compare_by_value():
    toks, _ = label_page("$x = .228$", "$x = 0.228$")
    assert all(t["op"] == "match" for t in toks)


def test_misread_is_number_to_number_only():
    gold = r"$V = 3 \cdot 4 = 12$ $\begin{aligned} I = 2 \end{aligned}$"
    target = r"$V = 8 \cdot 4 = 12$ $I = 2$ $R = 5$"
    toks, deleted = label_page(gold, target)
    by_raw = {t["raw"]: t for t in toks}
    assert by_raw["8"]["misread"] == 1 and by_raw["8"]["gold"] == "3.0"
    assert by_raw["12"]["misread"] == 0 and by_raw["12"]["op"] == "match"
    assert by_raw["5"]["op"] == "ins" and by_raw["5"]["misread"] == 0  # extra content is not a misread
    assert deleted == 0


def test_number_never_substitutes_for_a_variable():
    toks, deleted = label_page("$v_2 = 10$", "$40 = 10$")
    assert toks[0]["op"] == "ins" and toks[0]["misread"] == 0 and deleted == 1


def test_aggregate_takes_weakest_piece_and_builds_alternative():
    answer = "$I = 72$"
    # model pieces: '$', 'I', ' =', ' ', '7', '2', '$'  (digits are single tokens in Qwen)
    spans = [(0, 1), (1, 2), (2, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    pieces = [{"start": s, "end": e, "lp": -0.01, "gap": 0.9, "ent": 0.1, "dent": 0.0, "alt_piece": "x"}
              for s, e in spans]
    pieces[5].update(lp=-1.2, gap=-0.2, ent=1.5, dent=0.9, alt_piece="8")  # the '2' is doubtful: reader prefers 8
    rows = {r["text"]: r for r in aggregate(answer, pieces)}
    r = rows["72"]
    assert r["n_pieces"] == 2 and r["nll"] == 1.2 and r["neg_gap"] == 0.2 and r["dent"] == 0.9
    assert r["alt"] == "78"


def test_metrics():
    y = np.array([1, 0, 1, 0, 0])
    assert auroc(y, np.array([0.9, 0.1, 0.8, 0.2, 0.3])) == 1.0
    assert auroc(y, np.zeros(5)) == 0.5
    assert average_precision(y, np.array([0.9, 0.1, 0.8, 0.2, 0.3])) == 1.0
    df = pd.DataFrame({"page": ["a", "a", "a", "b", "b"], "misread": y, "s": [0.9, 0.1, 0.2, 0.3, 0.8]})
    assert at_k(df, "s", 1) == (0.5, 0.5)  # page a flags its misread, page b flags a false alarm


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL PASSED")
