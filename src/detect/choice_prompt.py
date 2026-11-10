"""The forced-choice question, shared by the verifier (verify_choice.py), the checker fine-tuning
(src/finetune/train_checker.py) and its evaluation (src/finetune/eval_checker.py), so the trained checker sees exactly
the question it is later asked, encoded the same way. No heavy imports."""
from __future__ import annotations

CONFUSE = {"0": "68", "1": "72", "2": "7", "3": "85", "4": "9", "5": "63", "6": "05", "7": "12", "8": "30", "9": "4"}
QUESTION = ("Look only at the student's handwriting in the image. In this line of their work:\n\n{line}\n\n"
            "the number marked [?] is written either as\nA) {a}\nB) {b}\n"
            "Which one did the student actually write, even if it is mathematically wrong? Answer with A or B.")


def masked_line(answer: str, start: int, end: int, width: int = 80) -> str:
    """The number's line (clipped to ~width chars each side) with the number replaced by [?]."""
    lo = answer.rfind("\n", 0, start) + 1
    hi = answer.find("\n", end)
    hi = len(answer) if hi < 0 else hi
    lo, hi = max(lo, start - width), min(hi, end + width)
    return answer[lo:start] + "[?]" + answer[end:hi]


def encode(proc, image, line: str, a: str, b: str):
    """Processor inputs for the question, ending where the model writes its answer (the generation prompt).
    The answer is read from (or trained on) the next token: 'A' or 'B'."""
    q = QUESTION.format(line=line, a=a, b=b)
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return proc(text=[text], images=[image], return_tensors="pt")


def ab_token_ids(tokenizer) -> list[int]:
    return [tokenizer.convert_tokens_to_ids(x) for x in ("A", "B")]
