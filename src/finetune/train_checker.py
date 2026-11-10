"""Fine-tune the A/B checker: Qwen2.5-VL-3B (QLoRA) answers "is the number marked [?] written as A) x or B) y?"
with the option that is WRITTEN on the page, including mathematically wrong values (pairs: ab_data.py).

Loss is on one token only: the answer letter, predicted from the last prompt position, which is exactly what
verify_choice.py reads (P('A') vs P('B')). The question is built by choice_prompt.encode, shared with the verifier.

Start point: the base model, or (--init-adapter) our verbatim reader's LoRA, trained further on the A/B task.

  python train_checker.py --pairs results/checker/pairs --sources fermat,edu --out results/checker/checker_fe \\
      --init-adapter local_results/results/reader_lora

Runs on one 11 GB GPU (2080 Ti: fp16 + 4-bit). Resumes from the last checkpoint in --out.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src" / "detect"))
from choice_prompt import ab_token_ids, encode  # noqa: E402


def read_pairs(folder: Path, sources: list[str], split: str) -> list[dict]:
    out = []
    for s in sources:
        f = folder / f"pairs_{s}_{split}.jsonl"
        if not f.exists():
            raise SystemExit(f"{f} missing: run ab_data.py {s} first")
        out += [json.loads(line) for line in f.open(encoding="utf-8")]
    return out


def balance(pairs: list[dict], max_examples: int | None, seed: int = 0) -> list[dict]:
    """At most max_examples, shared out evenly over the groups (source/kind, e.g. fermat/silent, edu/misread); a group
    with fewer pairs hands its unused share to the others. So the plentiful FERMAT look-alike pairs cannot swamp the
    scarce EDU misread pairs. Both orders of a pair stay together."""
    rng = random.Random(seed)
    groups: dict[tuple, list[dict]] = {}
    for q in pairs:  # a pair = its two orders (consecutive in the file, same line / written / options)
        key = (q["source"], q["kind"], q.get("fermat_idx"), q.get("page"), q["line"], q["written"], frozenset((q["a"], q["b"])))
        groups.setdefault(key, []).append(q)
    by_kind: dict[str, list[list[dict]]] = {}
    for g in groups.values():
        by_kind.setdefault(f"{g[0]['source']}/{g[0]['kind']}", []).append(g)
    for v in by_kind.values():
        rng.shuffle(v)
    if max_examples:  # water-filling: equal shares, a small group's unused share goes to the rest
        spare, take = max_examples // 2, {k: 0 for k in by_kind}  # in pairs
        while spare > 0:
            open_ = [k for k in sorted(by_kind) if take[k] < len(by_kind[k])]
            if not open_:
                break
            share = max(1, spare // len(open_))
            for k in open_:
                extra = min(share, len(by_kind[k]) - take[k], spare)
                take[k] += extra
                spare -= extra
        by_kind = {k: v[: take[k]] for k, v in by_kind.items()}
    out = [q for v in by_kind.values() for g in v for q in g]
    rng.shuffle(out)
    return out


def kind_counts(pairs: list[dict]) -> dict:
    c: dict[str, int] = {}
    for q in pairs:
        c[f"{q['source']}/{q['kind']}"] = c.get(f"{q['source']}/{q['kind']}", 0) + 1
    return c


class ImageSource:
    """FERMAT images by row index (dataset loaded once), EDU page images by path."""

    def __init__(self, need_fermat: bool):
        self.ds = None
        if need_fermat:
            from fermat_data import load_fermat
            self.ds = load_fermat()

    def __call__(self, q: dict):
        from PIL import Image
        if q["source"] == "fermat":
            return self.ds[int(q["fermat_idx"])]["image"].convert("RGB")
        return Image.open(q["image"]).convert("RGB")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--pairs", default="results/checker/pairs")
    p.add_argument("--sources", default="fermat", help="comma-separated: fermat, edu")
    p.add_argument("--init-adapter", default=None, help="start from this LoRA (e.g. the verbatim reader)")
    p.add_argument("--out", required=True)
    p.add_argument("--max-examples", type=int, default=6000, help="training examples (each pair counts twice)")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-pixels", type=int, default=768 * 28 * 28, help="same as the verifier's (LMA_REREAD_MAX_PIXELS)")
    p.add_argument("--device", type=int, default=0)
    a = p.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import Dataset
    from transformers import (AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, Trainer,
                              TrainingArguments)

    sources = [s for s in a.sources.split(",") if s]
    train = balance(read_pairs(Path(a.pairs), sources, "train"), a.max_examples)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = {"sources": sources, "init_adapter": a.init_adapter, "examples": len(train), "kinds": kind_counts(train),
            "epochs": a.epochs, "lr": a.lr, "rank": a.rank, "max_pixels": a.max_pixels}
    (out / "train_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))

    gpu = torch.cuda.is_available()
    bf16 = gpu and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    quant = gpu and not os.environ.get("LMA_NO_QUANT")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model, dtype=dtype if gpu else torch.float32, attn_implementation="sdpa",
        device_map={"": a.device if gpu else "cpu"},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        if quant else None)
    if quant:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if a.init_adapter:
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    # min_pixels as in run_ocr.HFReader, which the verifier uses: the checker sees images the same way
    proc = AutoProcessor.from_pretrained(a.model, min_pixels=256 * 28 * 28, max_pixels=a.max_pixels)
    ab = ab_token_ids(proc.tokenizer)
    images = ImageSource("fermat" in sources)

    class Pairs(Dataset):
        def __len__(self):
            return len(train)

        def __getitem__(self, i):
            return train[i]

    def collate(batch):
        q = batch[0]  # batch size 1: pages differ in size
        enc = dict(encode(proc, images(q), q["line"], q["a"], q["b"]))  # ends at the generation prompt
        enc["target"] = torch.tensor([ab[0] if q["answer"] == "A" else ab[1]])
        return enc

    class OneTokenTrainer(Trainer):
        """Loss on the answer letter only, from the logits at the last prompt position: exactly what
        verify_choice.py reads. logits_to_keep=1 avoids a full-vocabulary float copy of every position
        (~0.5 GB per page), which ran a 2080 Ti out of memory."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            target = inputs.pop("target")
            out = model(**inputs, logits_to_keep=1)
            loss = torch.nn.functional.cross_entropy(out.logits[:, -1].float(), target.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    args = TrainingArguments(
        output_dir=str(out), per_device_train_batch_size=1, gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=0.05,
        logging_steps=10, save_strategy="steps", save_steps=100, save_total_limit=2,
        bf16=bf16, fp16=gpu and not bf16, use_cpu=not gpu,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, dataloader_num_workers=0, report_to="none", seed=0)
    resume = any(out.glob("checkpoint-*"))
    trainer = OneTokenTrainer(model=model, args=args, train_dataset=Pairs(), data_collator=collate)
    trainer.model_accepts_loss_kwargs = False  # the Trainer then averages the loss over gradient accumulation itself
    trainer.train(resume_from_checkpoint=resume or None)
    model.save_pretrained(str(out))
    proc.save_pretrained(str(out))
    print(f"saved checker adapter to {out}")


if __name__ == "__main__":
    main()
