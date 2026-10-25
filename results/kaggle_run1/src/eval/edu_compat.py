"""Load the EDU-CIRCUIT-HW evaluation code *unchanged* and route its LLM calls through llm.call_llm.

We import their modules (utils/judge_ocr_diff.py, step_10, step_12) so that every prompt
and parsing rule is byte-for-byte theirs. Only the transport function `LLM_judge_content`
is replaced, so any judge model can be plugged in and calls are cached.

Side effects in their code that we neutralise:
  * judge_ocr_diff prints GOOGLE_API_KEY at import  -> we hide the variable during import
  * step_10 / step_12 call os.chdir at import         -> we restore the working directory
  * they import the deprecated `google.generativeai`  -> stubbed if not installed (unused)
"""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

from llm import Cache, call_llm


def _stub_missing(module_name: str, **attrs):
    try:
        importlib.import_module(module_name)
    except Exception:
        mod = types.ModuleType(module_name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[module_name] = mod


def load_edu_modules(edu_repo: str | os.PathLike):
    eval_dir = Path(edu_repo) / "check_recognition_error_val"
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Not an EDU-CIRCUIT-HW repo: {edu_repo}")
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))

    class _Any:  # placeholder enum-like object
        def __getattr__(self, name):
            return name

    _stub_missing("google.generativeai", configure=lambda **kw: None, GenerativeModel=None)
    _stub_missing("google.generativeai.types", HarmCategory=_Any(), HarmBlockThreshold=_Any())
    _stub_missing("openai", OpenAI=lambda **kw: None)
    _stub_missing("dotenv", load_dotenv=lambda *a, **kw: False)

    cwd = os.getcwd()
    hidden = os.environ.pop("GOOGLE_API_KEY", None)
    try:
        judge = importlib.import_module("utils.judge_ocr_diff")
        step10 = importlib.import_module("step_10_exclude_strict_items_in_OCR_err")
        step12 = importlib.import_module("step_12_recognition_error_taxonomy")
    finally:
        os.chdir(cwd)
        if hidden is not None:
            os.environ["GOOGLE_API_KEY"] = hidden
    return judge, step10, step12


def install_backend(modules, judge_model: str, cache: Cache | None):
    """Replace their LLM transport in every loaded module.

    The `model_name` they pass is ignored in favour of `judge_model`, so the whole
    pipeline uses one consistent judge (default: gemini-2.5-pro, as in the paper).
    """

    def LLM_judge_content(prompt, model_name=None):  # signature matches theirs
        return call_llm(prompt, judge_model, temperature=0.0, cache=cache)

    for m in modules:
        m.LLM_judge_content = LLM_judge_content
    return LLM_judge_content
