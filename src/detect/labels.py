"""Per-token misread labels: align each page of a model transcript with the expert-rectified page (Phase 0).

Ground truth for the misread detector. Only maths is compared ($..$, $$..$$, \\[..\\], \\(..\\)), with the same
token classes as eval/transcript_diff.py (numbers, operators, variables), but every token keeps its CHARACTER
OFFSETS in the target transcript, so a model's token probabilities can be attached to it (score_tokens.py).

Per target maths token:
  op       match | sub (gold has a different token of the SAME type here) | ins (no counterpart in gold)
  gold     the aligned gold token for 'sub'
  err      op != match
  misread  a number aligned to a DIFFERENT number (sub, num -> num): the clean label for the misread detector.
           'ins' numbers are mostly alignment noise or looping, so the detector is scored on misread vs match.

Formatting commands (\\frac, \\boxed, \\begin{aligned}, ...) are dropped and a substitution across token types costs
more than delete + insert, so a number is never 'misread' as a LaTeX keyword.

Pages are paired by file name ({qid}[_k]_markdown.md); a question whose page split differs is skipped and counted.

  python labels.py --target-root <Outputs/qwen3b_ft> --model-dir qwen3b_ft --out results/detect/qwen3b_ft
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "eval"))
from evaluate_recognition import RECTIFIED  # noqa: E402

ITEMS = ROOT / "src" / "eval" / "items_with_expert_transcript.csv"
GOLD_MODEL_DIR = "gemini-2.5-pro"

_MATH = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\[(.+?)\\\]|\\\((.+?)\\\)", re.S)
# order matters: \text{..} is prose, commands before bare letters, numbers keep thousands commas
_TOK = re.compile(r"(?P<skip>\\(?:text|mathrm|operatorname|begin|end|label|tag)\{[^}]*\}|\\[,;!])"
                  r"|(?P<cmd>\\[A-Za-z]+)"
                  r"|(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
                  r"|(?P<var>[A-Za-z]+(?:_\{?[A-Za-z0-9]+\}?)?)"
                  r"|(?P<op>[=+\-*/^<>]|Ω)")
_CMD = {"\\cdot": ("op", "*"), "\\times": ("op", "*"), "\\div": ("op", "/"), "\\Omega": ("var", "ohm")}
_GREEK = {"alpha", "beta", "gamma", "delta", "Delta", "epsilon", "theta", "lambda", "mu", "pi", "rho", "sigma",
          "tau", "phi", "Phi", "omega", "infty"}


def math_tokens(text: str) -> list[dict]:
    """Maths tokens with offsets: [{"i", "text" (normalised), "raw", "type", "start", "end"}]."""
    out = []
    for m in _MATH.finditer(text or ""):
        g = next(k for k in range(1, 5) if m.group(k) is not None)
        base, body = m.start(g), m.group(g)
        for t in _TOK.finditer(body):
            kind, raw = t.lastgroup, t.group()
            if kind == "skip":
                continue
            norm = raw
            if kind == "cmd":
                if raw in _CMD:
                    kind, norm = _CMD[raw]
                elif raw[1:] in _GREEK:
                    kind, norm = "var", raw[1:]
                else:  # formatting (\frac, \boxed, \int, arrows ...): not something a reader misreads as a value
                    continue
            elif kind == "num":  # by value: '.228' == '0.228', '72,000' == '72000' (but '05' != '.05')
                norm = repr(float(raw.replace(",", "")))
            elif kind == "var":
                norm = re.sub(r"[{}]", "", raw)
            elif raw == "Ω":
                kind, norm = "var", "ohm"
            out.append({"i": len(out), "text": norm, "raw": raw, "type": kind,
                        "start": base + t.start(), "end": base + t.end()})
    return out


def _cost(g: tuple[str, str], t: tuple[str, str]) -> int:
    """Substitution cost of (text, type) pairs: 0 equal, 1 same type, 3 across types (> delete + insert)."""
    return 0 if g[0] == t[0] else (1 if g[1] == t[1] else 3)


def align(gold: list[tuple[str, str]], target: list[tuple[str, str]]) -> list[tuple[str, int | None]]:
    """Levenshtein alignment over (text, type); for each TARGET token: (op, index of aligned gold token or None)."""
    n, m = len(gold), len(target)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        gi, row, prev = gold[i - 1], d[i], d[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + _cost(gi, target[j - 1]))
    ops: list = [None] * m
    i, j = n, m
    while j > 0:
        if i > 0 and d[i][j] == d[i - 1][j - 1] + _cost(gold[i - 1], target[j - 1]):
            ops[j - 1] = ("match" if gold[i - 1][0] == target[j - 1][0] else "sub", i - 1)
            i, j = i - 1, j - 1
        elif d[i][j] == d[i][j - 1] + 1:
            ops[j - 1] = ("ins", None)
            j -= 1
        else:  # gold token missing from the target: nothing to attach it to
            i -= 1
    return ops


def label_page(gold_text: str, target_text: str) -> tuple[list[dict], int]:
    """Target tokens with labels, plus the number of gold tokens the target dropped."""
    g, t = math_tokens(gold_text), math_tokens(target_text)
    ops = align([(x["text"], x["type"]) for x in g], [(x["text"], x["type"]) for x in t])
    for tok, (op, gi) in zip(t, ops):
        tok.update(op=op, gold=g[gi]["text"] if op == "sub" else "", err=int(op != "match"),
                   misread=int(op == "sub" and tok["type"] == "num"))
    deleted = len(g) - sum(1 for op, _ in ops if op != "ins")
    return t, deleted


def page_files(base: Path, hw: str, st: str, qid: str, model_dir: str) -> dict[str, Path]:
    folder = base / f"Homework_collected_database_trial_{hw}_{st}" / "models" / model_dir / "Compare"
    rx = re.compile(rf"^{re.escape(qid)}(_[1-9])?_markdown\.md$")
    return {f.name[: -len("_markdown.md")]: f for f in sorted(folder.glob(f"{qid}*_markdown.md")) if rx.match(f.name)}


def page_text(path: Path) -> str:
    """Page markdown without the image-link first line (as the transcriber's output, and as evaluate_recognition)."""
    return "\n".join(path.read_text(encoding="utf-8").split("\n")[1:])


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--edu-data", default=os.environ.get("EDU_DATA"))
    p.add_argument("--target-root", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    a = p.parse_args()

    items = pd.read_csv(ITEMS, dtype=str)[["Homework ID", "Student ID", "Question ID"]].values.tolist()
    rows, pages, skipped = [], [], 0
    for hw, st, qid in items[: a.limit] if a.limit else items:
        gold = page_files(Path(a.edu_data) / RECTIFIED, hw, st, qid, GOLD_MODEL_DIR)
        tgt = page_files(Path(a.target_root), hw, st, qid, a.model_dir)
        if not tgt or set(gold) != set(tgt):
            skipped += 1
            continue
        for stem, path in tgt.items():
            toks, deleted = label_page(page_text(gold[stem]), page_text(path))
            pid = f"{hw}|{st}|{stem}"
            pages.append({"page": pid, "student": st, "n_tokens": len(toks), "n_err": sum(t["err"] for t in toks),
                          "n_deleted": deleted})
            rows += [{"page": pid, "student": st, **tok} for tok in toks]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "labels.csv", index=False)
    pd.DataFrame(pages).to_csv(out / "label_pages.csv", index=False)
    num = df[df["type"] == "num"]
    summary = {"pages": len(pages), "questions_skipped_page_mismatch": skipped, "tokens": len(df),
               "token_err_rate": round(df["err"].mean(), 4), "num_tokens": len(num),
               "num_err": int(num["err"].sum()), "num_err_rate": round(num["err"].mean(), 4),
               "num_misread": int(num["misread"].sum()), "num_match": int((num["op"] == "match").sum()),
               "num_ins": int((num["op"] == "ins").sum()),
               "num_misread_rate_vs_match": round(num["misread"].sum() / max(1, (num["op"] != "ins").sum()), 4)}
    (out / "labels_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
