"""Transcribe EDU-CIRCUIT-HW pages with ANY model, writing outputs in the paper's layout
so `evaluate_recognition.py` (and their own scripts) can read them:

  Outputs/<task>/Homework_collected_database_trial_<HW>_<STUDENT>/models/<model_dir>/Compare/<qid>[_k]_markdown.md
  (first line is an image link, exactly as their script writes it)

Backends:
  --model gemini-2.5-pro                 Google API (their setup; T=0)
  --model openai:<name>                  any OpenAI-compatible API / local vLLM server
  --model hf:Qwen/Qwen2.5-VL-3B-Instruct local transformers (Kaggle/Ada GPU); --adapter <lora_dir> for our fine-tuned reader

Examples:
  python run_ocr.py --split debug --task repro_debug --model gemini-2.5-pro
  python run_ocr.py --split observation --task qwen3b_base --model hf:Qwen/Qwen2.5-VL-3B-Instruct
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import Cache, call_llm  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EDU_REPO = os.environ.get("EDU_REPO", str(PROJECT_ROOT / "external" / "CIRCUIT_EDU_HW_ACL"))
SPLIT_CSV = {"observation": "obsetf_involved_data.csv", "test": "test_involved_data.csv",
             "debug": "debug_involved_data.csv"}
DEFAULT_PROMPT = Path("handwritting_recognition_utils") / "Prompts" / "Initial_prompt_wo_problem_statement_v6.txt"


def normalize_code(s: str) -> str:  # same as theirs
    m = re.match(r"^(\d+_\d+-\d+)(?:_\d+)?$", s)
    return m.group(1) if m else s


def sanitize(filename: str) -> str:  # same as theirs
    return filename.replace(".", "_").replace("(", "").replace(")", "").replace("_png", "")


class HFReader:
    """Local Qwen2.5-VL reader (greedy decoding). Optional LoRA adapter for our fine-tuned model."""

    def __init__(self, model_id: str, adapter: str | None, max_pixels: int, max_new_tokens: int,
                 repetition_penalty: float = 1.0, device: int = 0):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype if torch.cuda.is_available() else torch.float32, attn_implementation="sdpa",
            device_map={"": device if torch.cuda.is_available() else "cpu"})  # one GPU per process (see --shard)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.proc = AutoProcessor.from_pretrained(model_id, min_pixels=256 * 28 * 28, max_pixels=max_pixels)
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty

    def __call__(self, prompt: str, image: Image.Image) -> str:
        import torch
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      repetition_penalty=self.repetition_penalty)
        return self.proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    def read_with_conf(self, prompt: str, image: Image.Image, max_new_tokens: int = 96) -> tuple[str, float]:
        """Short greedy read + confidence = geometric-mean probability of the generated tokens (agent re-reads)."""
        import torch
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      return_dict_in_generate=True, output_scores=True)
        new = out.sequences[:, inputs.input_ids.shape[1]:]
        logp = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
        keep = new[0] != self.proc.tokenizer.pad_token_id
        conf = float(torch.exp(logp[keep].float().mean())) if keep.any() else 0.0
        return self.proc.batch_decode(new, skip_special_tokens=True)[0].strip(), conf


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--edu-repo", default=DEFAULT_EDU_REPO)
    p.add_argument("--data-root", default=os.environ.get("EDU_DATA"), help="where Screenshot_output_anon lives (default: edu repo)")
    p.add_argument("--split", choices=list(SPLIT_CSV), default="debug")
    p.add_argument("--task", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model-dir", default=None, help="folder name under models/ (default: last part of model id)")
    p.add_argument("--prompt-file", default=None, help="default: their Initial_prompt_wo_problem_statement_v6.txt")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir (hf backend)")
    p.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--repetition-penalty", type=float, default=1.05,
                   help="stops small models looping on long pages until max-new-tokens (minutes per page)")
    p.add_argument("--shard", default=None, help="i/n: process every n-th page starting at i (one per GPU)")
    p.add_argument("--items", default=None, help="CSV (Homework ID, Student ID, Question ID) restricting the split")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--outputs-root", default=None, help="where to write Outputs/ (default: <data-root>/Outputs)")
    args = p.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        load_dotenv(Path(args.edu_repo) / ".env")
    except ImportError:
        pass

    edu, data = Path(args.edu_repo), Path(args.data_root or args.edu_repo)
    shots = data / "Screenshot_output_anon"
    allowed = pd.read_csv(shots / "set_splitting" / SPLIT_CSV[args.split], dtype=str)
    allowed = set(zip(allowed["Homework ID"], allowed["Student ID"], allowed["Question ID"]))
    if args.items:
        only = pd.read_csv(args.items, dtype=str)
        allowed &= set(zip(only["Homework ID"], only["Student ID"], only["Question ID"]))
    prompt = (Path(args.prompt_file) if args.prompt_file else edu / DEFAULT_PROMPT).read_text(encoding="utf-8").strip()
    prompt = prompt.replace("{problem_statement}", "")
    model_dir = args.model_dir or args.model.split(":", 1)[-1].split("/")[-1]

    jobs = []
    for img in sorted(shots.glob("Homework*/student_*/*.png")):
        hw, st = img.parent.parent.name, img.parent.name
        san = sanitize(img.name)
        if (hw, st, normalize_code(san)) in allowed:
            jobs.append((hw, st, san, img))
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        jobs = jobs[i::n]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} images in split '{args.split}' -> Outputs/{args.task}/.../models/{model_dir}/Compare/")

    reader = None
    if args.model.startswith("hf:"):
        reader = HFReader(args.model[3:], args.adapter, args.max_pixels, args.max_new_tokens, args.repetition_penalty)
    cache = Cache(PROJECT_ROOT / "results" / "llm_cache.sqlite")

    failures = 0
    deadline = float(os.environ.get("LMA_DEADLINE", "0") or 0)  # set by scripts/run_all.py --hours
    for hw, st, san, img_path in tqdm(jobs, desc="transcribing"):
        if deadline and time.time() > deadline:
            print("time budget reached: stopping early (re-run to continue)", file=sys.stderr)
            break
        out_dir = Path(args.outputs_root or data / "Outputs") / args.task / f"Homework_collected_database_trial_{hw}_{st}" / "models" / model_dir / "Compare"
        out_md = out_dir / f"{san}_markdown.md"
        if out_md.exists() and not args.overwrite:
            continue
        image = Image.open(img_path).convert("RGB")
        text = reader(prompt, image) if reader else call_llm(prompt, args.model, images=[image], temperature=0.0, cache=cache)
        if not text or text.startswith("Error:"):
            failures += 1
            print(f"FAILED {hw}/{st}/{san}: {text[:200] if text else 'empty'}", file=sys.stderr)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        image.save(out_dir / f"{san}_source.png")
        out_md.write_text(f"![Source Image]({san}_source.png)\n" + text, encoding="utf-8")
    print(f"Done. failures={failures}")


if __name__ == "__main__":
    main()
