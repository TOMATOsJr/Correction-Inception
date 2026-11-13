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


def test_number_beam_search_on_a_fake_reader():
    """expand() driven like BeamReader.continuations, with a fake next-character distribution."""
    import math
    from beam_candidates import NUM_CHARS, expand
    truth = {"": {"5": 0.9, "1": 0.1}, "5": {"9": 0.6, ".": 0.25, "4": 0.15}, "59": {}, "5.": {"9": 1.0},
             "5.9": {}, "54": {}, "1": {"2": 1.0}, "12": {}}

    def dist(txt):  # log P per NUM_CHARS and log P(end)
        d = truth.get(txt, {})
        lp = [math.log(d[c]) if d.get(c) else -1e9 for c in NUM_CHARS]
        rest = 1 - sum(d.values())
        return lp, math.log(rest) if rest > 1e-9 else -1e9
    beams, finished = [("", 0.0)], []
    for _ in range(6):
        rows = [dist(t) for t, _ in beams]
        keep, done = expand(beams, [r[0] for r in rows], [r[1] for r in rows], 4)
        finished += done
        if not keep:
            break
        beams = [(t, sc) for sc, _, t, _ in keep if sc > -1e8]
        if not beams:
            break
    finished.sort(key=lambda x: -x[1])
    got = [(t, round(math.exp(sc), 3)) for t, sc in finished if sc > -1e8]
    assert got[:4] == [("59", 0.54), ("5.9", 0.225), ("54", 0.135), ("12", 0.1)], got


def test_expand_number_grammar():
    from beam_candidates import NUM_CHARS, expand
    flat = [0.0] * len(NUM_CHARS)
    keep, done = expand([("", 0.0)], [flat], [0.0], 99)
    assert {t for _, _, t, _ in keep} == set("0123456789.") and not done  # no ',' first, nothing to finish
    keep, done = expand([("5.", 0.0)], [flat], [0.0], 99)
    assert {t for _, _, t, _ in keep} == {f"5.{d}" for d in "0123456789"} and not done
    keep, done = expand([("5.2", 0.0)], [flat], [0.0], 99)
    assert "5.2." not in {t for _, _, t, _ in keep} and "5.2," not in {t for _, _, t, _ in keep} and done


def test_arbiter_thresholds():
    import numpy as np
    from arbiter import budget_threshold, net_threshold
    p = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.5])
    good = np.array([1, 1, 0, 1, 0, 0], bool)
    bad = np.array([0, 0, 1, 0, 1, 1], bool)
    assert net_threshold(p, good, bad) == 0.9          # +2 at 0.9; +2 again at 0.7 but the higher threshold wins
    assert net_threshold(p, good & False, bad) == 1.01  # never change when nothing nets positive
    broken = np.array([0, 0, 1, 0, 1, 1], bool)
    assert budget_threshold(p, broken, n_correct=100, max_fc=0.01) == 0.7  # one broken allowed
    assert budget_threshold(p, broken, n_correct=10, max_fc=0.01) == 0.9   # none allowed


def test_arbiter_evidence_and_heldout_decisions():
    import numpy as np
    import pandas as pd
    from arbiter import CHANGE, evidence, labels_of, learn_actions
    from selector import student_folds
    rng = np.random.default_rng(0)
    rows = []
    for k in range(400):
        mis = k % 4 == 0
        good = mis and k % 8 == 0
        pc = rng.uniform(0.8, 1.0) if good else rng.uniform(0.0, 0.85)
        best = "54" if good else "58"
        rows.append({"page": f"p{k // 10}", "student": f"s{(k // 4) % 8}", "i": k, "text": "59", "gold": "54" if mis else "59",
                     "misread": int(mis), "best": best, "p_change": pc, "consistent": True,
                     "p_cands": f"{best}:{pc:.3f}|50:0.1", "p_blank": pc / 3, "beams": "54|58", "neg_gap": 0.5,
                     "cand_is_key": False, "orig_is_key": False})
    df = pd.DataFrame(rows)
    X = evidence(df)
    assert X.loc[0, "in_beams"] == 1 and X.loc[0, "beam_rank"] == 0 and X.loc[1, "beam_rank"] == 1
    assert X.loc[0, "one_digit"] == 1 and abs(X.loc[0, "margin"] - (df.loc[0, "p_change"] - 0.1)) < 1e-3
    good, wrong, broken = labels_of(df)
    assert good.sum() == 50 and wrong.sum() == 50 and broken.sum() == 300
    res = learn_actions(df, X, student_folds(df["student"], 4))
    ch = res["actions"]["net"] == CHANGE
    assert (ch & good).sum() > 30 and (ch & ~good).sum() < 10   # separable: mostly the good changes


def test_fermat_audit_labels():
    from fermat_audit import label_sample
    pert = r"x = 2 + 3 = \textcolor{red}{6} \Rightarrow y = 12"
    orig = r"x = 2 + 3 = 5 \Rightarrow y = 12"
    kinds = {r["raw"]: r["kind"] for r in label_sample(0, "x = 2 + 3 = 5 \\Rightarrow y = 13", pert, orig, "g0")}
    assert kinds == {"2": "ok", "3": "ok", "5": "silent", "13": "misread"}
    keep = {r["raw"]: (r["kind"], r["gold"]) for r in label_sample(0, "x = 2 + 3 = 6 \\Rightarrow y = 12", pert, orig, "g0")}
    assert keep["6"] == ("ok", "6")  # the student's error preserved: what the Auditor wants


def test_consistency_features_and_rules():
    import numpy as np
    import pandas as pd
    from consistency import allowed_mask, consistency_features, decide, page_numbers
    page = "$R = 54$ ohm and $V = 108$ so $I = 108 / 59 = 2$ then $P = 54 \\cdot 2$"
    nums = page_numbers(page)
    i59 = next(i for i, v in nums if v == 59.0)
    df = pd.DataFrame([{"page": "p", "i": i59, "best": "54", "text": "59"},
                       {"page": "p", "i": i59, "best": "58", "text": "59"}])
    f = consistency_features(df, {"p": page})
    assert f.loc[0, "best_elsewhere"] == 2 and f.loc[0, "written_elsewhere"] == 0 and f.loc[0, "best_later"] == 1
    assert f.loc[1, "best_elsewhere"] == 0
    assert list(allowed_mask("required", f)) == [True, False] and list(allowed_mask("veto", f)) == [True, True]
    # decisions: the main rule never changes a number whose challenger is not elsewhere on the page
    rows = []
    for k in range(80):
        good = k % 4 == 0
        rows.append({"page": "p", "student": f"s{(k // 4) % 4}", "i": i59, "text": "59", "best": "54" if good else "58",
                     "gold": "54" if good else "59", "misread": int(good), "p_change": 0.99 if good else 0.6,
                     "consistent": True, "p_blank": 0.1, "cand_is_key": False, "orig_is_key": False})
    d2 = pd.DataFrame(rows)
    f2 = consistency_features(d2, {"p": page})
    act, _ = decide(d2, f2, np.array([k % 4 for k in range(80)]), "required", 0.01)
    assert all(a == "keep" for a, b in zip(act, d2["best"]) if b == "58") and any(a == "change" for a in act)


def test_edu_silent_categories():
    import pandas as pd
    from edu_silent import categorise
    lab = pd.DataFrame([
        {"page": "H|s1|3_6-1", "type": "num", "op": "sub", "raw": "70", "gold_raw": "72", "misread": 1},  # toward key
        {"page": "H|s1|3_6-1", "type": "num", "op": "sub", "raw": "13", "gold_raw": "12", "misread": 1},  # away from key
        {"page": "H|s1|3_6-1", "type": "num", "op": "sub", "raw": "9", "gold_raw": "8", "misread": 1},    # neither
        {"page": "H|s1|3_6-1", "type": "num", "op": "match", "raw": "70", "gold_raw": "", "misread": 0},
        {"page": "H|s1|3_6-1", "type": "num", "op": "ins", "raw": "5", "gold_raw": "", "misread": 0}])
    df = categorise(lab, {"3_6-1": {70.0, 12.0}})
    assert list(df["category"]) == ["toward_key", "away_key", "other", "ok"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ALL PASSED")
