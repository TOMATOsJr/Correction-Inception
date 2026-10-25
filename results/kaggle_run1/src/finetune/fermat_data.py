"""FERMAT utilities: problem-level splits, training pairs, and the silent-correction (error-reversion) metric.

FERMAT rows: image (handwritten erroneous solution), pert_a (what is WRITTEN, incl. the planted error),
orig_a (the correct solution), orig_q (problem), has_error, handwriting_style, ...

Split: by problem (orig_q) so no problem appears in two splits: ~1650 / 200 / 400 solutions (train/dev/test).
Training target: pert_a (verbatim, errors preserved). ALL rows are used, not only erroneous ones, so the
model learns "copy what is written" rather than "always insert a mistake".

Error-reversion (silent correction) metric, per erroneous sample:
  locate the spans where pert_a differs from orig_a; take a small context window around each span;
  if the model output matches the ORIGINAL (correct) window better than the PERTURBED (written) window,
  the reader silently corrected the student -> reverted. Rate = reverted / evaluable samples.
"""
from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher

PROMPT = ("Transcribe the student's handwritten solution exactly as written, in LaTeX/Markdown. "
          "Copy every step verbatim, INCLUDING any mistakes. Do not correct, complete or improve anything.")

_TOK = re.compile(r"\\[A-Za-z]+|\d+\.?\d*|[A-Za-z]+|[^\sA-Za-z\d]")


def tokens(s: str) -> list[str]:
    return _TOK.findall(s or "")


def norm(s: str) -> str:
    return " ".join(tokens(s))


def split_of(orig_q: str, seed: str = "lma2026") -> str:
    """Deterministic problem-level split (~73.5/9/17.5 % of problems -> ~1650/200/400 solutions)."""
    h = int(hashlib.sha256(f"{seed}|{orig_q}".encode()).hexdigest(), 16) % 1000
    return "train" if h < 735 else ("dev" if h < 825 else "test")


def add_splits(ds):
    return ds.map(lambda r: {"split": split_of(r["orig_q"])})


def _best_ratio(window: list[str], hay: list[str]) -> float:
    """Best similarity of `window` against any same-length slice of `hay` (fuzzy partial match)."""
    if not window or not hay:
        return 0.0
    n, best = len(window), 0.0
    w = " ".join(window)
    for i in range(0, max(1, len(hay) - n + 1)):
        best = max(best, SequenceMatcher(None, w, " ".join(hay[i:i + n])).ratio())
        if best == 1.0:
            break
    return best


def reversion(output: str, pert_a: str, orig_a: str, ctx: int = 3):
    """Return True (silently corrected), False (error preserved) or None (not evaluable)."""
    p, o, out = tokens(pert_a), tokens(orig_a), tokens(output)
    ops = [op for op in SequenceMatcher(None, o, p, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not ops or not out:
        return None
    votes = []
    for _, i1, i2, j1, j2 in ops:
        ow = o[max(0, i1 - ctx): i2 + ctx]
        pw = p[max(0, j1 - ctx): j2 + ctx]
        so, sp = _best_ratio(ow, out), _best_ratio(pw, out)
        if abs(so - sp) < 0.02:  # cannot tell (e.g. region not transcribed at all)
            continue
        votes.append(so > sp)
    if not votes:
        return None
    return sum(votes) > len(votes) / 2


def cer(output: str, ref: str) -> float:
    """Token-level error rate of output vs reference (1 - matched/len(ref))."""
    r, h = tokens(ref), tokens(output)
    if not r:
        return 0.0
    m = sum(b.size for b in SequenceMatcher(None, r, h, autojunk=False).get_matching_blocks())
    return 1 - m / max(len(r), len(h))


if __name__ == "__main__":  # tiny self-test
    orig = r"x = 2 + 3 = 5 \Rightarrow y = 12 / 5"
    pert = r"x = 2 + 3 = 6 \Rightarrow y = 12 / 6"
    assert reversion(r"x = 2 + 3 = 5 \Rightarrow y = 12 / 5", pert, orig) is True
    assert reversion(r"x=2+3=6 \Rightarrow y = 12/6", pert, orig) is False
    assert split_of("q1") == split_of("q1")
    print("fermat_data self-test passed")
