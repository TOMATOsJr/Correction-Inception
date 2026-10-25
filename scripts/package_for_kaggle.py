"""Build the two zips to upload as PRIVATE Kaggle Datasets:
  lma_src.zip                 -> our code (src/)
  edu_circuit_hw_images.zip   -> only the EDU-CIRCUIT-HW page images in debug/observation/test splits + splits + rubrics
Usage:  python scripts/package_for_kaggle.py [--data D:/lma/data/EDU-CIRCUIT-HW_v1] [--out D:/lma/kaggle_upload]
"""
import argparse
import re
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def qid_of(filename: str) -> str:
    san = filename.replace(".", "_").replace("(", "").replace(")", "").replace("_png", "")
    m = re.match(r"^(\d+_\d+-\d+)(?:_\d+)?$", san)
    return m.group(1) if m else san


def package_text(data: Path, out: Path):
    """edu_text.zip: everything the judge/grader needs, no images.
    edu_data/  -> expert transcripts, model transcripts (markdown only), splits, rubrics
    edu_repo/  -> the paper's evaluation code, one-shot example and shipped judge verdicts"""
    repo = ROOT / "external" / "CIRCUIT_EDU_HW_ACL"
    n = 0
    with zipfile.ZipFile(out / "edu_text.zip", "w", zipfile.ZIP_DEFLATED) as z:
        globs = [("Rectified_recognized_markdown_done_Anon", "**/*_markdown.md"),
                 ("Observationset_Final", "v6_*/**/Compare/*_markdown.md"),
                 ("Screenshot_output_anon", "set_splitting/*.csv"),
                 ("Screenshot_output_anon", "rubric_outputs/*.json")]
        for base, pattern in globs:
            for f in (data / base).glob(pattern):
                z.write(f, f"edu_data/{f.relative_to(data).as_posix()}")
                n += 1
        ev = repo / "check_recognition_error_val"
        sel = ev / "LLM-as-a-judge-selection"
        files = (list(ev.glob("*.py")) + list((ev / "utils").glob("*.py")) + list((sel / "fewshot_example_v2").glob("*"))
                 + list((sel / "0_finaloutput_rechecked").glob("*.csv"))
                 + list((sel / "0_finaloutput_rechecked_taxonomy").glob("*.csv")))
        for f in files:
            if f.is_file():
                z.write(f, f"edu_repo/{f.relative_to(repo).as_posix()}")
                n += 1
    print(f"edu_text.zip: {n} files")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="D:/lma/data/EDU-CIRCUIT-HW_v1")
    p.add_argument("--out", default="D:/lma/kaggle_upload")
    p.add_argument("--text-only", action="store_true", help="only build edu_text.zip and lma_src.zip")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    package_text(Path(a.data), out)
    with zipfile.ZipFile(out / "lma_src.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in (ROOT / "src").rglob("*"):
            if f.is_file() and f.suffix in (".py", ".txt", ".csv") and "__pycache__" not in f.parts:
                z.write(f, f"src/{f.relative_to(ROOT / 'src').as_posix()}")
        for f in ["scripts/run_all.py", "requirements-gpu.txt"]:  # one-shot runner (Kaggle smoke test)
            z.write(ROOT / f, f)
    if a.text_only:
        return
    shots, out = Path(a.data) / "Screenshot_output_anon", Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    keep = set()
    for f in ["obsetf", "test", "debug"]:
        s = pd.read_csv(shots / "set_splitting" / f"{f}_involved_data.csv", dtype=str)
        keep |= set(zip(s["Homework ID"], s["Student ID"], s["Question ID"]))
    n = 0
    with zipfile.ZipFile(out / "edu_circuit_hw_images.zip", "w", zipfile.ZIP_STORED) as z:
        for img in shots.glob("Homework*/student_*/*.png"):
            if (img.parent.parent.name, img.parent.name, qid_of(img.name)) in keep:
                z.write(img, f"Screenshot_output_anon/{img.relative_to(shots).as_posix()}")
                n += 1
        for f in list((shots / "set_splitting").glob("*.csv")) + list((shots / "rubric_outputs").glob("*.json")):
            z.write(f, f"Screenshot_output_anon/{f.relative_to(shots).as_posix()}")
    with zipfile.ZipFile(out / "lma_src.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in (ROOT / "src").rglob("*.py"):
            z.write(f, f"src/{f.relative_to(ROOT / 'src').as_posix()}")
    print(f"{n} images packaged")
    for f in out.iterdir():
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
