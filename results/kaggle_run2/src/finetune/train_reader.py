"""QLoRA fine-tuning of the faithful (verbatim) reader: Qwen2.5-VL-3B on FERMAT train (image -> pert_a).

Run on a GPU (Kaggle T4 x1 is enough, ~16 GB):
  python train_reader.py --out /kaggle/working/reader_lora --epochs 2
Loss is computed only on the transcript tokens (prompt + image tokens are masked).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fermat_data import PROMPT, add_splits  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--out", default="reader_lora")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-pixels", type=int, default=768 * 28 * 28, help="caps visual tokens (memory)")
    p.add_argument("--max-target-chars", type=int, default=3000, help="skip extremely long targets")
    p.add_argument("--limit", type=int, default=None, help="debug: train on first N examples")
    a = p.parse_args()

    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, Trainer,
                              TrainingArguments)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    ds = add_splits(load_dataset("ai4bharat/FERMAT", split="train", token=os.environ.get("HF_TOKEN")))
    train = ds.filter(lambda r: r["split"] == "train" and len(r["pert_a"] or "") <= a.max_target_chars)
    if a.limit:
        train = train.select(range(min(a.limit, len(train))))
    print(f"train examples: {len(train)} (all rows, erroneous and clean)")

    proc = AutoProcessor.from_pretrained(a.model, min_pixels=128 * 28 * 28, max_pixels=a.max_pixels)
    proc.tokenizer.padding_side = "right"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model, torch_dtype=dtype, attn_implementation="sdpa", device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    def collate(batch):
        texts, prompts, images = [], [], []
        for ex in batch:
            user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}
            asst = {"role": "assistant", "content": [{"type": "text", "text": ex["pert_a"]}]}
            texts.append(proc.apply_chat_template([user, asst], tokenize=False))
            prompts.append(proc.apply_chat_template([user], tokenize=False, add_generation_prompt=True))
            images.append(ex["image"].convert("RGB"))
        enc = proc(text=texts, images=images, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i in range(len(batch)):  # mask prompt + image tokens
            plen = proc(text=[prompts[i]], images=[images[i]], return_tensors="pt")["input_ids"].shape[1]
            labels[i, :plen] = -100
        enc["labels"] = labels
        return enc

    args = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=1, gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, save_strategy="steps", save_steps=50, save_total_limit=2, bf16=bf16, fp16=not bf16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, dataloader_num_workers=2, report_to="none", seed=0)
    resume = any(Path(a.out).glob("checkpoint-*")) if Path(a.out).exists() else False
    Trainer(model=model, args=args, train_dataset=train, data_collator=collate).train(resume_from_checkpoint=resume or None)
    model.save_pretrained(a.out)
    proc.save_pretrained(a.out)
    print(f"saved LoRA adapter to {a.out}")


if __name__ == "__main__":
    main()
