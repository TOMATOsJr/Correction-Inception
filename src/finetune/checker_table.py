"""Tables for the checker run (scripts/run_checker.sh): A/B accuracy per model, and the correction outcome.

  python checker_table.py --eval results/checker/eval                       # accuracy table
  python checker_table.py --eval results/checker/eval --best                # print the best checker's tag
  python checker_table.py --eval results/checker/eval --correct results/checker   # + correction table
Writes <run folder>/CHECKER_RESULTS.md (numbers only, safe to commit).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

KINDS = [("edu/misread", "EDU misread: picks the true value"),
         ("edu/lookalike", "EDU correct number: keeps it"),
         ("fermat/silent", "FERMAT wrong value: keeps the written one"),
         ("fermat/lookalike", "FERMAT correct number: keeps it")]
ORDER = ["base", "reader", "checker_f", "checker_fe"]


def best_checker(summary: dict) -> str:
    """The trained checker with the highest EDU misread accuracy; ties broken by EDU look-alike accuracy."""
    cands = [t for t in summary if t.startswith("checker")]
    if not cands:
        raise SystemExit("no trained checker evaluated yet")

    def key(t):
        k = summary[t]
        return (k.get("edu/misread", {}).get("accuracy", 0), k.get("edu/lookalike", {}).get("accuracy", 0))
    return max(sorted(cands), key=key)


def accuracy_table(summary: dict) -> str:
    tags = [t for t in ORDER if t in summary] + sorted(t for t in summary if t not in ORDER)
    head = "| Pairs | " + " | ".join(tags) + " |\n|---|" + "---|" * len(tags) + "\n"
    rows = []
    for k, label in KINDS:
        cells = []
        for t in tags:
            v = summary[t].get(k)
            cells.append(f"{100 * v['accuracy']:.1f}% ({100 * v['both_orders']:.0f}% both orders, n={v['pairs']})"
                         if v else "–")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def correction_table(run: Path) -> str:
    rows = []
    for d in sorted(run.glob("correct_*")):
        cm = d / "correct_metrics.json"
        if not cm.exists():
            continue
        h = json.loads(cm.read_text())["heldout"]
        diff = {}
        for v in ("none", "corrected"):
            f = d / "transcript_diff" / f"{d.name[len('correct_'):]}_{v}" / "metrics.json"
            if f.exists():
                diff[v] = json.loads(f.read_text()).get("mean_number_mismatches")
        vm = d / "verify_metrics.json"
        v = json.loads(vm.read_text()) if vm.exists() else {}
        pct = (lambda x: f"{100 * x:.0f}%" if isinstance(x, (int, float)) else "–")
        rows.append(f"| {d.name[len('correct_'):]} | {h['verified']} | {h['verified_misreads']} | "
                    f"{pct(v.get('candidate_recall'))} | {pct(v.get('beam_recall'))} | **{h['auto_fixed']}** | "
                    f"{h['auto_wrong_fix']} | **{h['auto_broken_correct']}** | {h['misreads_before']} → "
                    f"{h['misreads_after_auto']} | {diff.get('none', '–')} → {diff.get('corrected', '–')} |")
    if not rows:
        return ""
    return ("| Run | Numbers checked | of them misreads | True value among options | …among beams | Fixed | "
            "Changed to another wrong value | Correct numbers broken | Misreads before → after | "
            "Wrong numbers per exam before → after |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")


def accuracy_rows_present(summary: dict) -> str:
    """accuracy_table without rows nobody has (e.g. the EDU rows of a FERMAT-only evaluation)."""
    t = accuracy_table(summary).splitlines()
    keep = t[:2] + [r for r in t[2:] if any(c.strip() not in ("–", "") for c in r.split("|")[2:-1])]
    return "\n".join(keep) + "\n"


def arbiter_table(run: Path) -> str:
    rows = []
    for f in sorted(run.glob("correct_*/arbiter/arbiter_metrics.json")):
        m = json.loads(f.read_text())
        tag = f.parent.parent.name[len("correct_"):]
        ev = m.get("evidence_auroc", {})
        single = ev.get("single_evidence (direction-free)", {})
        for rule in ("net", "budget"):
            r = m[rule]
            rows.append(f"| {tag} | {rule} | {r['fixable']} | **{r['auto_fixed']}** | {r['auto_wrong_fix']} | "
                        f"{r['auto_broken_correct']} | **{r['net_fixes']:+d}** | {r['misreads_before']} → "
                        f"{r['misreads_after_auto']} | {ev.get('arbiter_heldout', '–')} | {single.get('p_change', '–')} |")
    if not rows:
        return ""
    return ("| Run | Rule | Fixable (true value offered) | Fixed | Changed to another wrong value | Correct broken | "
            "Net | Misreads before → after | Arbiter AUROC | Checker alone AUROC |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")


def consistency_table(run: Path) -> str:
    rows = []
    for f in sorted(run.glob("correct_*/consistency/consistency_metrics.json")):
        m = json.loads(f.read_text())
        tag = f.parent.parent.name[len("correct_"):]
        diff = {}
        for v in ("none", "corrected"):
            g = f.parent / "transcript_diff" / f"{tag}_consistency_{v}" / "metrics.json"
            if g.exists():
                diff[v] = json.loads(g.read_text()).get("mean_number_mismatches")
        for rule in ("simple", "required", "veto", "consistency_only"):
            if rule not in m:
                continue
            r = m[rule]
            dd = f"{diff.get('none', '–')} → {diff.get('corrected', '–')}" if rule == "required" else ""
            rows.append(f"| {tag} | {rule}{' (main)' if rule == 'required' else ''} | **{r['auto_fixed']}** | "
                        f"{r['auto_wrong_fix']} | {r['auto_broken_correct']} | **{r['net_fixes']:+d}** | "
                        f"{r['misreads_before']} → {r['misreads_after_auto']} | {dd} |")
        dg = m.get("diagnostics", {}).get("patterns", {}).get("challenger elsewhere and written not")
        if dg:
            rows.append(f"| {tag} | *pattern share: fixing vs bad changes* | {dg['share_of_fixing_changes']} | | "
                        f"{dg['share_of_bad_changes']} | | | |")
    if not rows:
        return ""
    return ("| Run | Rule | Fixed | Changed to another wrong value | Correct broken | Net | Misreads before → after | "
            "Wrong numbers per exam before → after |\n|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")


def edu_silent_section(run: Path) -> str:
    out = []
    for f in sorted(run.glob("edu_silent/*/edu_silent_metrics.json")):
        m = json.loads(f.read_text())
        c = m["categories"]
        out.append(f"**{m['name']}**: {m['numbers']} numbers on {m['pages']} pages, {m['misreads']} misread. Of the "
                   f"misreads, **{c.get('toward_key', 0)} towards the answer key** (transcript shows a key value, the "
                   f"student wrote another: silent-correction-like), {c.get('away_key', 0)} away from it (costing marks), "
                   f"{c.get('other', 0)} neither. Base rate: {100 * (m['base_rate_correct_numbers_in_key'] or 0):.1f}% of "
                   f"correctly transcribed numbers are key values; transcript value in the key for "
                   f"{100 * (m['misreads_whose_transcript_value_is_in_key'] or 0):.1f}% of misreads, written value for "
                   f"{100 * (m['misreads_whose_written_value_is_in_key'] or 0):.1f}%.")
        for k, v in (m.get("blank_image") or {}).items():
            if isinstance(v, dict):
                out.append(f"- {k}: " + ", ".join(f"{c2} {r['direction_free']}{' (low)' if r['direction'] == 'low' else ''}"
                                                  for c2, r in v.items()))
        out.append("")
    return "\n".join(out)


def audit_section(run: Path) -> str:
    root = Path(__file__).resolve().parents[2]
    out = []
    reader = []
    for m in ("qwen3b_base", "qwen3b_ft"):
        f = root / "results" / "pc_run" / "results" / "fermat" / m / "metrics_clean.json"
        old = f.with_name("metrics.json")
        if f.exists():
            c, o = json.loads(f.read_text()), json.loads(old.read_text()) if old.exists() else {}
            reader.append(f"| {m} | {o.get('silent_correction_rate_percent', '–')}% | **{c['silent_correction_rate_percent']}%** | "
                          f"{o.get('token_error_rate_vs_written_percent', '–')}% | **{c['token_error_rate_vs_written_percent']}%** |")
    if reader:
        out += ["Whole answers (FERMAT test). The old numbers left the red error markers in the written text, so preserved "
                "errors counted as silent corrections:\n",
                "| Reader | Silent correction (old) | Silent correction (markers stripped) | Token error (old) | Token error (stripped) |",
                "|---|---|---|---|---|", *reader, ""]
    for d in sorted(list(run.glob("audit_fermat_*")) + list(run.glob("audit_fix_*"))):
        ls, dm = d / "labels_summary.json", d / "detect_metrics.json"
        if not ls.exists():
            continue
        lab = json.loads(ls.read_text())
        out.append(f"**{d.name.replace('audit_fermat_', '').replace('audit_fix_', 'fix step, ')}**: {lab['numbers']} numbers, {lab['kinds']}; planted-error spots "
                   f"{lab['error_spot_numbers']}, silently corrected {100 * lab['silent_share_of_error_spots']:.1f}%.")
        if dm.exists():
            m = json.loads(dm.read_text())
            for kind in ("silent", "misread"):
                a = m.get(f"auroc_{kind}_vs_ok")
                if a:
                    top = sorted(((k, v) for k, v in a.items() if k != "learned_detector"), key=lambda kv: -abs(kv[1] - 0.5))[:3]
                    out.append(f"- {kind} vs ok: learned detector AUROC **{a['learned_detector']}**; strongest single signals "
                               "(direction-free; *low* = smaller values mean the error) "
                               + ", ".join(f"{k} {max(v, 1 - v):.3f}{' (low)' if v < 0.5 else ''}" for k, v in top)
                               + f"; caught in the top {m['k_per_sample']} per answer: {m[f'{kind}_caught_in_top_k']}"
                                 f" of {m[f'{kind}_total']}")
        am = d / "audit_metrics.json"
        if am.exists():
            amj = json.loads(am.read_text())
            for rule, name in (("simple", "simple rule"), ("net", "Arbiter, net rule")):
                if rule not in amj:
                    continue
                r = amj[rule]
                out.append(f"- fixes ({name}): silent corrections {r['silent_before']} → {r['silent_after']} "
                           f"(true value offered for {r['silent_fixable']} of the {r['silent_checked']} checked), misreads "
                           f"fixed {r['misread_fixed']}, changed to another wrong value {r['auto_wrong_fix']}, correct numbers "
                           f"broken {r['auto_broken_correct']}; answers silently corrected "
                           f"{r['silent_correction_rate_before_percent']}% → {r['silent_correction_rate_after_percent']}%")
        out.append("")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", required=True)
    p.add_argument("--best", action="store_true")
    p.add_argument("--correct", default=None, help="run folder holding correct_<tag>/")
    a = p.parse_args()
    sp = Path(a.eval) / "summary.json"
    summary = json.loads(sp.read_text()) if sp.exists() else {}
    if a.best:
        print(best_checker(summary))
        return
    run = Path(a.correct) if a.correct else Path(a.eval).parent
    md = ["# A/B checker results\n", "Generated by `src/finetune/checker_table.py`. Plan: `docs/CHECKER.md`.\n",
          "## 1. Does the checker pick what is written? (held-out pairs)\n",
          "Accuracy = share of pairs where the right option gets more than half the probability, averaged over both "
          "option orders. Chance is 50%. The fix only pays off if *EDU misread* is well above 50% while *EDU correct "
          "number* stays near 100%.\n", accuracy_table(summary) if summary else "(no evaluations yet)\n"]
    ct = correction_table(run)
    if ct:
        md += ["\n## 2. Automatic correction with the checker (unseen students; `_k10_beams` = 10 numbers per page with "
               "beam candidates, otherwise the 3 most doubtful)\n",
               "Success = more *Fixed* than *Correct numbers broken* + *Changed to another wrong value*, and fewer "
               "wrong numbers per exam after than before.\n", ct]
    cs = Path(a.eval).parent / "eval_clean" / "summary.json"
    if cs.exists():
        md += ["\n## 3. FERMAT pairs without the red error markers\n",
               "The first FERMAT pairs kept FERMAT's `\\textcolor{red}{..}` error markers in the shown line (46% of the "
               "silent test pairs had one right at [?]), a shortcut. Same models, markers removed (and many more silent "
               "pairs, since the markers had also broken the alignment):\n", accuracy_rows_present(json.loads(cs.read_text()))]
    at = arbiter_table(run)
    if at:
        md += ["\n## 4. Learned Arbiter: all tool evidence combined (unseen students)\n",
               "Checker probability and margin, order agreement, blank-image probability, beam rank and probability, "
               "reader doubt, edit shape and answer-key flags in one logistic model, trained on other students. *net*: "
               "threshold with the most net fixes on the training students (no change unless positive); *budget*: at most "
               "1% of correct numbers changed.\n", at]
    au = audit_section(run)
    if au:
        md += ["\n## 5. The Auditor on FERMAT (silent corrections, exact labels)\n", au]
    cons = consistency_table(run)
    if cons:
        md += ["\n## 6. Consistency with the student's own work (unseen students)\n",
               "Does the challenger (or the written value) appear elsewhere on the page? *required* is the main rule, "
               "fixed in advance: change only when the challenger does and the written value does not, plus the usual "
               "checker conditions and threshold (<= 1% of correct numbers changed on the other students).\n", cons]
    es = edu_silent_section(run)
    if es:
        md += ["\n## 7. Silent corrections in the real exams (EDU, via the answer key)\n", es]
    (run / "CHECKER_RESULTS.md").write_text("\n".join(md))
    sys.stdout.write("\n".join(md))


if __name__ == "__main__":
    main()
