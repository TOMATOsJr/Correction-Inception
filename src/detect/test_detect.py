"""Tests for misread labels and the detection metrics (run: python src/detect/test_detect.py). No model needed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_detection import at_k, auroc, average_precision  # noqa: E402
from labels import label_page, math_tokens  # noqa: E402
from score_tokens import aggregate  # noqa: E402
from sweep_correct import digit_doubt, with_guards  # noqa: E402
from verify_choice import crop_band, candidates, decide, key_numbers, masked_line, metrics, other_values, qid_of  # noqa: E402
from selector import LogReg, cutoff_for_recall, student_folds  # noqa: E402
from correct import CHANGE, KEEP, TA, actions, pick_tau, summarise, write_variants  # noqa: E402


def test_selector_pieces():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    y = (X[:, 0] + 0.1 * rng.normal(size=400) > 0.8).astype(int)
    p = LogReg().fit(X, y).predict(X)
    assert auroc(y, p) > 0.95  # learns the informative feature
    s, yy = np.array([0.9, 0.8, 0.7, 0.6, 0.1]), np.array([1, 0, 1, 1, 0])
    assert cutoff_for_recall(s, yy, 0.66) == 0.7 and cutoff_for_recall(s, yy, 1.0) == 0.6
    f = student_folds(pd.Series(["b", "a", "c", "a"]), 2)
    assert f[1] == f[3] and len(set(f)) == 2  # one fold per student


def test_verify_helpers():
    rub = {"notes": "get 72,000 C; p = 1100 W (1.1 kW); cost $0.264", "criteria": [{"deduction_rules": "uses 7200 s"}]}
    assert {72000.0, 1100.0, 1.1, 0.264, 7200.0} <= key_numbers(rub)
    assert candidates("59", "58|57", 3, first=("54",)) == ["54", "58", "57"]  # other transcriber's value first
    assert other_values("$V = 54 + 2$", "$V = 59 + 2$") == {2: "54"}
    assert qid_of("1_5-2") == "1_5-2" and qid_of("1_5-2_2") == "1_5-2"


def _verified():
    return pd.DataFrame({  # 59 -> 54 strong; 12 correct, weak challenger; 7 correct but key guard; 3 -> 8 flips
        "page": ["h|s1|q"] * 4, "student": ["s1"] * 4, "i": [2, 4, 6, 8], "start": [5, 10, 15, 20],
        "end": [7, 12, 16, 21], "text": ["59", "12", "7", "3"], "gold": ["54.0", "", "", "8.0"],
        "misread": [1, 0, 0, 1], "best": ["54", "17", "1", "8"], "p_change": [0.9, 0.2, 0.95, 0.7],
        "consistent": [True, True, True, False], "p_blank": [0.5, np.nan, 0.5, np.nan],
        "cand_is_key": [False, False, True, False], "orig_is_key": [False, False, False, False]})


def test_actions_and_guards():
    df = _verified()
    assert list(actions(df, 0.8, 0.35, 0.15)) == [CHANGE, KEEP, TA, TA]
    df.loc[0, "p_blank"] = 0.85  # the reader prefers 54 almost as much without the page: guessing, not reading
    assert actions(df, 0.8, 0.35, 0.15)[0] == TA
    assert pick_tau(df[df["misread"] == 1], 0.35, 0.15, 0.01) == 1.01  # nothing to tune on: never auto-change


def test_summarise_and_write_variants(tmp=None):
    import tempfile
    df = _verified()
    act = actions(df, 0.8, 0.35, 0.15)
    s = summarise(df, act, all_misreads=10, pages=1)
    assert s["auto_fixed"] == 1 and s["auto_broken_correct"] == 0 and s["ta_queue_misreads"] == 1
    assert s["misreads_after_auto"] == 9 and s["misreads_after_auto_and_perfect_ta"] == 8
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "src"
        src = root / "Homework_collected_database_trial_h_s1" / "models" / "m" / "Compare"
        src.mkdir(parents=True)
        body = "$V = 59 + 12 + 7 + 3$"
        (src / "q_markdown.md").write_text("![img](q_source.png)\n" + body, encoding="utf-8")
        df["start"] = [body.index(x) for x in ("59", "12", "7", "3")]
        df["end"] = df["start"] + df["text"].str.len()
        labels = pd.DataFrame({"page": ["h|s1|q"] * 2, "i": [2, 8], "gold_raw": ["54", "8"]})
        write_variants(Path(t), df, act, labels, root, "m")
        out = lambda v: (Path(t) / "Outputs" / v / "Homework_collected_database_trial_h_s1" / "models" / "m" /  # noqa: E731
                         "Compare" / "q_markdown.md").read_text(encoding="utf-8")
        assert out("none") == "![img](q_source.png)\n" + body
        assert out("corrected").endswith("$V = 54 + 12 + 7 + 3$")
        assert out("corrected_ta").endswith("$V = 54 + 12 + 7 + 8$")  # the TA fixes the flipping 3 -> 8


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
    assert r["alts"] == ""  # no digit candidates given


def test_aggregate_alts_swap_weakest_digit():
    answer = "$I = 72$"
    spans = [(0, 1), (1, 2), (2, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    pieces = [{"start": s, "end": e, "lp": -0.01, "gap": 0.9, "ent": 0.1, "dent": 0.0, "alt_piece": "x",
               "alt_digits": ""} for s, e in spans]
    pieces[4].update(dent=0.1, alt_digits="1")
    pieces[5].update(gap=-0.2, dent=0.9, alt_digits="837")  # the '2': reader's runner-up digits 8, 3, 7
    r = {r["text"]: r for r in aggregate(answer, pieces)}["72"]
    assert r["alts"] == "78|73|77"


def test_candidates_reader_first_then_lookalikes_no_duplicates():
    c = candidates("38", "35|38|36", 10)
    assert c == ["35", "36", "88", "58", "33", "30"]  # reader's (minus the reading itself), then look-alikes
    assert candidates("38", float("nan"), 2) == ["88", "58"]


def test_masked_line_and_decide():
    ans = "first line\n$V = 3 \\cdot 4$\nlast"
    s = ans.index("3")
    assert masked_line(ans, s, s + 1) == "$V = [?] \\cdot 4$"
    d = decide([("8", 0.7, True), ("6", 0.9, False)])
    assert d["best"] == "6" and not d["propose"]  # strongest challenger flips with the order: TA, not a fix
    assert decide([("8", 0.7, True), ("6", 0.2, True)])["propose"]
    assert not decide([])["propose"]


def test_verify_metrics():
    df = pd.DataFrame({"misread": [1, 1, 0, 0], "gold": ["3.0", "5.0", "", ""], "cands": ["3|6", "6", "1", "4"],
                       "best": ["3", "6", "1", "4"], "propose": [True, True, False, True],
                       "consistent": [True, True, True, False], "p_change": [0.9, 0.8, 0.1, 0.6],
                       "neg_gap": [0.5, 0.4, 0.3, 0.2]})
    m = metrics(df)
    assert m["candidate_recall"] == 0.5 and m["fix_rate"] == 0.5 and m["wrong_fix_rate"] == 0.5
    assert m["false_change_rate"] == 0.5 and m["auroc_p_change"] == 1.0


def test_crop_band_stays_inside_page():
    from PIL import Image
    img = Image.new("RGB", (100, 1000))
    assert crop_band(img, 0, 100, 0) is img
    assert crop_band(img, 0, 100, 0.3).size == (100, 300)  # first number: top of the page
    assert crop_band(img, 100, 100, 0.3).size == (100, 300)  # last number: bottom of the page


def test_sweep_guards_and_digit_filter():
    df = pd.DataFrame({"cand_is_key": [True], "alt": ["12"], "p_change": [0.9], "p_blank": [0.9]})
    assert with_guards(df, "no_key", 0.15)[0]["cand_is_key"].tolist() == [False]
    assert with_guards(df, "no_blank", 0.15)[1] < 0
    assert with_guards(df, "all", 0.15)[1] == 0.15
    assert digit_doubt(pd.DataFrame({"alt": ["12", " ", None]})).tolist() == [True, False, False]


def test_metrics():
    y = np.array([1, 0, 1, 0, 0])
    assert auroc(y, np.array([0.9, 0.1, 0.8, 0.2, 0.3])) == 1.0
    assert auroc(y, np.zeros(5)) == 0.5
    assert average_precision(y, np.array([0.9, 0.1, 0.8, 0.2, 0.3])) == 1.0
    df = pd.DataFrame({"page": ["a", "a", "a", "b", "b"], "misread": y, "s": [0.9, 0.1, 0.2, 0.3, 0.8]})
    assert at_k(df, "s", 1) == (0.5, 0.5)  # page a flags its misread, page b flags a false alarm


def test_beam_candidates_parse_and_dedupe():
    from beam_candidates import distinct_candidates, leading_number
    assert leading_number("5.9 V = I R$") == "5.9" and leading_number(" 540}") == "540"
    assert leading_number("72,000 \\Omega") == "72,000" and leading_number(".25 A") == ".25"
    assert leading_number("$x") is None
    conts = [("59$", 0.5), ("59 V", 0.2), ("5.9$", 0.1), ("54$", 0.08), ("590", 0.05), ("x", 0.01), ("54.0", 0.03)]
    got = distinct_candidates(conts, "59", 4)
    assert [c for c, _ in got] == ["5.9", "54", "590"]  # written value dropped, 54 == 54.0 merged
    assert abs(got[1][1] - 0.11) < 1e-9
    assert len(distinct_candidates(conts, "59", 2)) == 2


def test_verify_metrics_beam_recall():
    import pandas as pd
    from verify_choice import metrics
    df = pd.DataFrame([
        {"misread": 1, "gold": "5.9", "beams": "5.9|54", "cands": "5.9|54", "best": "5.9", "propose": True,
         "consistent": True, "p_change": 0.9, "neg_gap": 0.5},
        {"misread": 1, "gold": "12", "beams": "123", "cands": "123|128", "best": "", "propose": False,
         "consistent": True, "p_change": 0.1, "neg_gap": 0.4},
        {"misread": 0, "gold": "", "beams": "", "cands": "1|7", "best": "", "propose": False,
         "consistent": True, "p_change": 0.2, "neg_gap": 0.1}])
    m = metrics(df)
    assert m["beam_recall"] == 0.5 and m["candidate_recall"] == 0.5 and m["fix_rate"] == 0.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL PASSED")
