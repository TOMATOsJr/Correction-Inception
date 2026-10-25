"""Unified LLM/VLM call layer with on-disk caching and retries.

Backends (chosen by model-name prefix):
  gemini-*  / models/gemini-*   -> Google `google-genai` SDK  (env: GEMINI_API_KEY or GOOGLE_API_KEY)
  openai:<model>                -> any OpenAI-compatible server (env: OPENAI_API_KEY, OPENAI_BASE_URL)
                                   e.g. openai:gpt-4o, or a local vLLM server serving Qwen
  hf-text:<hf id>               -> local text LLM via transformers, 4-bit (free judge fallback on Kaggle)
  mock                          -> deterministic fake, for self-tests (no network)

Errors are returned as strings starting with "Error:" to match the EDU-CIRCUIT-HW
convention (their step 11 counts such rows as 1 error); they are never cached.
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
import sqlite3
import threading
import time
from pathlib import Path

_SAFETY_CATEGORIES = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
]


class Cache:
    """Tiny thread-safe sqlite key/value cache, so re-runs never re-pay for API calls."""

    def __init__(self, path: str | os.PathLike | None):
        self.path = path
        self._lock = threading.Lock()
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(path), check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
            self._db.commit()

    def get(self, key: str):
        if not self.path:
            return None
        with self._lock:
            row = self._db.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str):
        if not self.path:
            return
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, value))
            self._db.commit()


def _image_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _cache_key(model: str, temperature: float, prompt: str, images) -> str:
    h = hashlib.sha256()
    h.update(f"{model}|{temperature}|".encode())
    h.update(prompt.encode("utf-8"))
    for img in images or []:
        h.update(hashlib.sha256(_image_bytes(img)).digest())
    return h.hexdigest()


# ---------------------------------------------------------------- backends
_gemini_client = None


def _call_gemini(model: str, prompt: str, images, temperature: float) -> str:
    global _gemini_client
    from google import genai
    from google.genai import types

    if _gemini_client is None:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment / .env")
        _gemini_client = genai.Client(api_key=key)
    config = types.GenerateContentConfig(
        temperature=temperature,
        safety_settings=[types.SafetySetting(category=c, threshold="BLOCK_NONE") for c in _SAFETY_CATEGORIES],
    )
    contents = [prompt] + list(images or [])
    resp = _gemini_client.models.generate_content(
        model=model.split("/")[-1], contents=contents, config=config
    )
    if not resp.text:
        raise ValueError("empty response text")
    return resp.text


def _call_openai(model: str, prompt: str, images, temperature: float) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )
    content = [{"type": "text", "text": prompt}]
    for img in images or []:
        b64 = base64.b64encode(_image_bytes(img)).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    resp = client.chat.completions.create(
        model=model.split(":", 1)[1],
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


_hf_text = {}
_hf_lock = threading.Lock()
HF_BATCH = int(os.environ.get("HF_BATCH", "8"))
# padded tokens per batch (longest prompt + 1024 new) x batch size; ~16k fits a 7B 4-bit model on a 15 GB T4
HF_TOKEN_BUDGET = int(os.environ.get("HF_TOKEN_BUDGET", "16000"))
MAX_NEW = 1024


def _hf_worker(mid: str, device: int, q):
    """One model copy per GPU; pulls queued prompts and generates them together, within a token budget.
    On CUDA OOM the batch is retried one prompt at a time; only a single prompt that still OOMs fails."""
    import queue

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(mid, padding_side="left")
    mdl = AutoModelForCausalLM.from_pretrained(
        mid, device_map={"": device}, dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                               bnb_4bit_quant_type="nf4"))

    def encode(p):
        return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)

    def generate(items):
        ids = tok([t for t, _, _ in items], return_tensors="pt", padding=True).to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**ids, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        for (_, _, box), row in zip(items, out):
            box["text"] = tok.decode(row[ids["input_ids"].shape[1]:], skip_special_tokens=True)
            box["done"].set()

    held = None  # an item that didn't fit the previous batch
    while True:
        p, box = held or q.get()
        held = None
        t = encode(p)
        batch = [(t, len(tok(t).input_ids), box)]
        longest = batch[0][1]
        while len(batch) < HF_BATCH:
            try:
                p2, b2 = q.get(timeout=0.5)
            except queue.Empty:
                break
            t2 = encode(p2)
            n2 = len(tok(t2).input_ids)
            if (max(longest, n2) + MAX_NEW) * (len(batch) + 1) > HF_TOKEN_BUDGET:
                held = (p2, b2)
                break
            batch.append((t2, n2, b2))
            longest = max(longest, n2)
        try:
            generate(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            for item in batch:
                try:
                    generate([item])
                except Exception as e:
                    torch.cuda.empty_cache()
                    item[2]["err"] = e
                    item[2]["done"].set()
        except Exception as e:
            for _, _, b in batch:
                b["err"] = e
                b["done"].set()


def _call_hf_text(model: str, prompt: str, temperature: float) -> str:
    """Local text-only LLM via transformers, batched across concurrent callers and all visible GPUs.
    model = 'hf-text:<hf id>'. Use many --workers so batches fill up."""
    import queue

    import torch

    mid = model.split(":", 1)[1]
    with _hf_lock:
        if mid not in _hf_text:
            q = queue.Queue()
            for d in range(max(1, torch.cuda.device_count())):
                threading.Thread(target=_hf_worker, args=(mid, d, q), daemon=True).start()
            _hf_text[mid] = q
    box = {"done": threading.Event()}
    _hf_text[mid].put((prompt, box))
    box["done"].wait()
    if "err" in box:
        raise box["err"]
    return box["text"]


def _call_mock(prompt: str) -> str:
    # Recognises the three EDU-CIRCUIT-HW prompt types by their wording.
    if "Respond with ONLY the word \"DELETE\"" in prompt:  # step 10 filter
        return "KEEP"
    if "Output ONLY the category number" in prompt:  # step 12 taxonomy
        return "1"
    if "teaching assistant grading" in prompt:  # grade_impact grader
        return '{"deductions": {}, "reasons": {}}'
    if "OCR (Optical Character Recognition) Quality Assurance Judge" in prompt:  # step 9 judge
        if "MOCK_CLEAN" in prompt.split("### CURRENT TASK:")[-1]:
            return "No significant errors found."
        return "1.\nSource: 8 A\nRectified Version: 3 A\nReason: Wrong value"
    return "mock transcript"


def call_llm(prompt: str, model: str, *, images=None, temperature: float = 0.0,
             cache: Cache | None = None, max_retries: int = 5) -> str:
    """Call `model` with a text prompt (+ optional PIL images). Returns text or 'Error: ...'."""
    if model == "mock":
        return _call_mock(prompt)
    key = _cache_key(model, temperature, prompt, images)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit
    last_err = None
    for attempt in range(max_retries):
        try:
            if model.startswith("openai:"):
                text = _call_openai(model, prompt, images, temperature)
            elif model.startswith("hf-text:"):
                text = _call_hf_text(model, prompt, temperature)
            elif "gemini" in model.lower():
                text = _call_gemini(model, prompt, images, temperature)
            else:
                raise ValueError(f"Unknown model backend for '{model}' (use gemini-*, openai:<name>, or mock)")
            if cache is not None:
                cache.set(key, text)
            return text
        except ValueError as e:
            if "Unknown model backend" in str(e):
                raise
            last_err = e
        except Exception as e:  # network, quota, safety block, ...
            last_err = e
            if any(t in str(e) for t in ("404", "NOT_FOUND", "API_KEY_INVALID", "PERMISSION_DENIED")):
                break  # permanent error: retrying won't help
        time.sleep(min(60, 2 ** attempt * 3))
    return f"Error: {last_err}"
