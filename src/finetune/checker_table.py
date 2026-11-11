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
    for d in sorted(run.glob("audit_fermat_*")):
        ls, dm = d / "labels_summary.json", d / "detect_metrics.json"
        if not ls.exists():
            continue
        lab = json.loads(ls.read_text())
        out.append(f"**{d.name[len('audit_fermat_'):]}**: {lab['numbers']} numbers, {lab['kinds']}; planted-error spots "
                   f"{lab['error_spot_numbers']}, silently corrected {100 * lab['silent_share_of_error_spots']:.1f}%.")
        if dm.exists():
            m = json.loads(dm.read_text())
            for kind in ("silent", "misread"):
                a = m.get(f"auroc_{kind}_vs_ok")
                if a:
                    top = sorted(((k, v) for k, v in a.items() if k != "learned_detector"), key=lambda kv: -abs(kv[1] - 0.5))[:3]
                    out.append(f"- {kind} vs ok: learned detector AUROC **{a['learned_detector']}**; strongest single signals "
                               + ", ".join(f"{k} {v}" for k, v in top)
                               + f"; caught in the top {m['k_per_sample']} per answer: {m[f'{kind}_caught_in_top_k']}"
                                 f" of {m[f'{kind}_total']}")
        am = d / "audit_metrics.json"
        if am.exists():
            r = json.loads(am.read_text())["net"]
            out.append(f"- fixes (Arbiter, net rule): silent corrections {r['silent_before']} → {r['silent_after']}, misreads "
                       f"fixed {r['misread_fixed']}, correct numbers broken {r['auto_broken_correct']}; answers silently "
                       f"corrected {r['silent_correction_rate_before_percent']}% → {r['silent_correction_rate_after_percent']}%")
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
    (run / "CHECKER_RESULTS.md").write_text("\n".join(md))
    sys.stdout.write("\n".join(md))


if __name__ == "__main__":
    main()
