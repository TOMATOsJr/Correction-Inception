"""Recompute the FERMAT metrics of existing predictions with the colour markers stripped (no GPU, no download).

The original metrics compared transcripts with pert_a still containing \\textcolor{red}{..}, so preserved errors were
scored as silent corrections. fermat_data.reversion / cer now strip the markers; this re-scores saved predictions.

  python src/finetune/rescore_fermat.py results/pc_run/results/fermat/qwen3b_base results/pc_run/results/fermat/qwen3b_ft
Writes metrics_clean.json next to each predictions.jsonl and prints a comparison with the old metrics.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fermat_data import cer, reversion  # noqa: E402


def rescore(d: Path) -> dict:
    recs = [json.loads(x) for x in (d / "predictions.jsonl").open(encoding="utf-8") if x.strip()]
    recs = [r for r in recs if not str(r["output"]).startswith("Error:")]
    ev = [v for v in (reversion(r["output"], r["pert_a"], r["orig_a"]) for r in recs if r["has_error"]) if v is not None]
    by_style: dict[str, list] = {}
    for r in recs:
        if r["has_error"]:
            v = reversion(r["output"], r["pert_a"], r["orig_a"])
            if v is not None:
                by_style.setdefault(str(r["handwriting_style"]), []).append(v)
    return {"name": d.name, "n": len(recs), "n_erroneous_evaluable": len(ev),
            "silent_correction_rate_percent": round(100 * sum(ev) / len(ev), 2) if ev else None,
            "token_error_rate_vs_written_percent": round(100 * sum(cer(r["output"], r["pert_a"]) for r in recs) / max(1, len(recs)), 2),
            "silent_correction_by_handwriting_style": {k: [round(100 * sum(v) / len(v), 1), len(v)] for k, v in by_style.items()},
            "note": "red error markers stripped from pert_a and output (see fermat_data.strip_red)"}


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        d = Path(arg)
        m = rescore(d)
        (d / "metrics_clean.json").write_text(json.dumps(m, indent=2))
        old = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
        print(f"{d.name}: silent correction {old.get('silent_correction_rate_percent')}% -> {m['silent_correction_rate_percent']}%,"
              f" token error {old.get('token_error_rate_vs_written_percent')}% -> {m['token_error_rate_vs_written_percent']}%")
