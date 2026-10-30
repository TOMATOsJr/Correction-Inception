"""Collect every ablation metric into one Markdown report: results/ablation/ABLATION_RESULTS.md.

  python scripts/collect_ablation.py --out results/ablation --transcribers qwen3b_ft gemini25

Reads, per transcriber T and run R:
  <out>/T/R/ablate_summary.json                Arbiter decisions, claim precision, verifier accuracy
  <out>/T/R/decide_summary.json                interaction + agent-tool stats (interactive/solo runs)
  <out>/T/R/transcript_diff/<name>/metrics.json   maths diff vs the expert transcript
  results/grade_impact/<name>/metrics.json        TCE (grade impact)          with <name> = <prefix>_T_R__<variant>
Missing pieces show as "-", so the report can be built at any point of the run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def fmt(x, nd=1):
    return "-" if x is None or x == "" else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


class Report:
    def __init__(self, out: Path, prefix: str = "abl"):
        self.out, self.prefix = out, prefix
        self.lines: list[str] = []

    def summary(self, t, run):
        return load(self.out / t / run / "ablate_summary.json")

    def metrics(self, t, run, variant) -> dict:
        name = f"{self.prefix}_{t}_{run}__{variant}"
        g = load(ROOT / "results" / "grade_impact" / name / "metrics.json").get("code_level", {})
        d = load(self.out / t / run / "transcript_diff" / name / "metrics.json")
        ci = g.get("TCE_95CI_student_bootstrap")
        return {"TCE": g.get("TCE_rate_percent"), "CI": f"{ci[0]:.1f}–{ci[1]:.1f}" if ci else None,
                "wrongful": g.get("wrongful_deduction_percent"), "over": g.get("over_credit_percent"),
                "numerr": d.get("number_error_sample_rate_percent"), "mism": d.get("mean_number_mismatches"),
                "n": g.get("n_samples") or d.get("n_samples")}

    def decisions(self, t, run, key) -> dict:
        return self.summary(t, run).get("configs", {}).get(key, {})

    def table(self, header: list[str], rows: list[list]):
        self.lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        self.lines += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
        self.lines.append("")

    def outcome_row(self, label, t, run, variant, dec_key=None):
        m = self.metrics(t, run, variant)
        d = self.decisions(t, run, dec_key) if dec_key else {}
        return [label, m["n"], m["TCE"], m["CI"], m["wrongful"], m["over"], m["numerr"], m["mism"],
                d.get("accept"), d.get("escalate"), d.get("reject"), d.get("precision_percent")]


OUTCOME = ["System / variant", "n", "TCE %", "95% CI", "Wrongful %", "Over-credit %", "Exams w/ number error %",
           "Mean number mismatches", "Accepted", "Escalated", "Rejected", "Claim precision %"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/ablation")
    p.add_argument("--prefix", default="abl", help="name prefix used by run_ablation.sh (LMA_GI_PREFIX)")
    p.add_argument("--transcribers", nargs="+", default=["qwen3b_ft", "gemini25"])
    a = p.parse_args()
    out = Path(a.out)
    R = Report(out, a.prefix)
    L = R.lines
    L += ["# Ablation results", "",
          "TCE = % of (exam, rubric code) pairs whose deduction differs between grading the expert transcript and "
          "grading this transcript (lower is better). Claim precision = helpful / (helpful + harmful) accepted claims, "
          "judged against the expert transcript. Accepted / Escalated / Rejected are Arbiter decisions over all exams. "
          "Unless stated otherwise: fine-tuned reader, disputed value masked (`ft_mask`), full evidence.", ""]

    for t in a.transcribers:
        s = R.summary(t, "interactive")
        theta = s.get("theta", 0.85)
        main = s.get("main_tag", "ft_mask")
        tk = f"full__{main}__t{theta}"
        L += [f"## Transcriber: `{t}`", f"Exams: {s.get('exams', '-')} · claims ruled on (interactive): {s.get('claims', '-')} · θ = {theta}", ""]

        L += ["### #9 Headline: system comparison", ""]
        R.table(OUTCOME, [
            R.outcome_row("No correction (`none`)", t, "interactive", "none"),
            R.outcome_row("B2 one-pass self-correction", t, "selfcorrect", "selfcorrect"),
            R.outcome_row("One-shot agents + Arbiter", t, "oneshot", f"full__{main}", tk),
            R.outcome_row("Interactive agents + Arbiter (verify pass)", t, "interactive", f"full__{main}", tk),
            R.outcome_row("Interactive, as built (agent-requested evidence only)", t, "interactive", "agenttools"),
        ])

        L += ["### #2 Evidence used by the Arbiter (interactive run)", ""]
        R.table(OUTCOME, [
            R.outcome_row("Re-read + consistency (full)", t, "interactive", f"full__{main}", tk),
            R.outcome_row("Re-read only", t, "interactive", f"reread_only__{main}", f"reread_only__{main}__t{theta}"),
            R.outcome_row("Consistency only", t, "interactive", f"consistency_only__{main}", f"consistency_only__{main}__t{theta}"),
            R.outcome_row("No verification (accept every claim)", t, "interactive", "noverify", "noverify"),
        ])

        L += ["### #3 Threshold sweep (interactive run, full evidence)", ""]
        rows = []
        for th in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
            d = R.decisions(t, "interactive", f"full__{main}__t{th}")
            variant = f"full__{main}" if th == theta else f"full__{main}__t{th}"
            m = R.metrics(t, "interactive", variant)
            rows.append([str(th), d.get("accept"), d.get("escalate"), d.get("reject"), d.get("helps"), d.get("hurts"),
                         d.get("precision_percent"), m["TCE"], m["numerr"]])
        R.table(["θ", "Accepted", "Escalated", "Rejected", "Accepted helpful", "Accepted harmful",
                 "Claim precision %", "TCE % (if scored)", "Exams w/ number error % (if scored)"], rows)

        L += ["### #6 Base vs fine-tuned reader · #7 Disputed value hidden vs shown (interactive run)", "",
              "Verifier accuracy = of the re-reads that picked either the transcribed or the proposed value, how often "
              "the pick agrees with the expert transcript.", ""]
        rows = []
        for tag in ("ft_mask", "base_mask", "ft_nomask", "base_nomask"):
            v = s.get("verifier", {}).get(tag, {})
            d = R.decisions(t, "interactive", f"full__{tag}__t{theta}")
            m = R.metrics(t, "interactive", f"full__{tag}")
            rows.append([tag, v.get("accuracy_vs_expert_percent"), v.get("evaluable"), v.get("picked_proposed"),
                         v.get("picked_span"), v.get("picked_unresolved"), v.get("no_reading"), v.get("mean_conf"),
                         d.get("accept"), d.get("precision_percent"), m["TCE"], m["numerr"]])
        R.table(["Reader config", "Verifier accuracy %", "Evaluable", "Picked proposed", "Picked transcribed",
                 "Unresolved", "No reading", "Mean conf", "Accepted", "Claim precision %", "TCE %",
                 "Exams w/ number error %"], rows)

        L += ["### #10 True single-agent control", "",
              "Post-hoc = that agent's claims taken out of the joint (interactive) run; solo = the agent ran alone.", ""]
        R.table(OUTCOME, [
            R.outcome_row("Advocate, post-hoc from joint run", t, "interactive", f"posthoc_advocate__{main}"),
            R.outcome_row("Advocate, solo run", t, "solo_advocate", f"full__{main}", f"full__{main}__t{theta}"),
            R.outcome_row("Auditor, post-hoc from joint run", t, "interactive", f"posthoc_auditor__{main}"),
            R.outcome_row("Auditor, solo run", t, "solo_auditor", f"full__{main}", f"full__{main}__t{theta}"),
            R.outcome_row("Both agents (interactive)", t, "interactive", f"full__{main}", tk),
        ])

        L += ["### Interaction and agent-tool stats", ""]
        rows = []
        for run in ("interactive", "solo_advocate", "solo_auditor"):
            d = load(out / t / run / "decide_summary.json")
            i, tl = d.get("interaction", {}), d.get("tools", {})
            rows.append([run, i.get("filed_r1"), i.get("filed_r2"), i.get("countered"), i.get("withdrawn"),
                         i.get("revised"), i.get("conflicts_after_r3"), i.get("rejected_actions"),
                         tl.get("crop_reread"), tl.get("ok"), tl.get("failed")])
        R.table(["Run", "Filed r1", "Filed r2", "Countered", "Withdrawn", "Revised", "Conflicts after r3",
                 "Rejected actions", "crop_reread calls", "Tool ok", "Tool failed"], rows)

    L += ["## #16 Transcript quality", "",
          "Compare each section above across transcribers. Caveat: the expert transcripts were made by correcting "
          "Gemini 2.5 Pro's output, which favours Gemini.", ""]
    R.table(["Transcriber"] + OUTCOME, [[t] + R.outcome_row(lbl, t, run, var, key) for t in a.transcribers
                                        for lbl, run, var, key in (
                                            ("none", "interactive", "none", None),
                                            ("interactive + Arbiter", "interactive", "full__ft_mask",
                                             f"full__ft_mask__t{R.summary(t, 'interactive').get('theta', 0.85)}"))])

    path = out / "ABLATION_RESULTS.md"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
