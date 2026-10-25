# Correction Inception: setup guide

Your laptop handles data, API calls and analysis. GPU work runs on Kaggle, Colab or Ada.
Legend: **[MANUAL]** = do it yourself in a browser · **[TERMINAL]** = copy-paste commands

---

## Part 1: accounts and keys [MANUAL], about 15 minutes
1. **Hugging Face:** sign up at huggingface.co → Settings → Access Tokens → create a **Read** token. Copy it.
2. **FERMAT access:** open https://huggingface.co/datasets/ai4bharat/FERMAT while logged in → accept the terms (the dataset is gated).
3. **Gemini API key:** https://aistudio.google.com → Get API key. Check the price of Gemini 2.5 Pro before large runs.
4. **Kaggle:** sign up → Settings → **verify your phone number**. This unlocks the free GPUs (T4, about 30 hours a week).
5. **Ada (if used):** log in and run `nvidia-smi` to see the GPU and its VRAM.
6. **EDU-CIRCUIT-HW data:** open https://doi.org/10.35090/gatech/81252 → download the zip into `D:\lma\`. It is a Georgia Tech Digital Library page, so the download is manual.

---

## Part 2: laptop setup [TERMINAL] (PowerShell)
```powershell
# 0. Install Git if missing (check with: git --version)
winget install --id Git.Git -e

# 1. Workspace on D: (plenty of space)
mkdir D:\lma; cd D:\lma

# 2. Clone the EDU-CIRCUIT-HW repo
git clone https://github.com/gt-learning-innovation/CIRCUIT_EDU_HW_ACL
cd CIRCUIT_EDU_HW_ACL

# 3. Unzip the dataset (from Part 1, step 6) into the repo root.
#    The zip contains 4 folders that must sit next to README.md. Replace <zipname>.
Expand-Archive D:\lma\<zipname>.zip -DestinationPath D:\lma\CIRCUIT_EDU_HW_ACL

# 4. Python environment
python -m venv D:\lma\env
D:\lma\env\Scripts\Activate.ps1
pip install --upgrade pip
pip install datasets huggingface_hub pandas openpyxl sympy python-dotenv tqdm
# Plus the repo's own requirements. Check its README for a requirements file, e.g.:
# pip install -r requirements.txt
```

**[MANUAL]** Open `D:\lma\CIRCUIT_EDU_HW_ACL\.env` in a text editor and set:
```
GEMINI_API_KEY=your_key_here
```
Never commit or share this file.

```powershell
# 5. Hugging Face login (paste the token from Part 1)
huggingface-cli login

# 6. Download FERMAT (4.79 GB) to D:
python -c "from datasets import load_dataset; ds=load_dataset('ai4bharat/FERMAT', cache_dir='D:/hf_cache'); print(ds)"

# 7. Smoke test of the EDU OCR on the small debug split (calls the Gemini API)
cd D:\lma\CIRCUIT_EDU_HW_ACL\handwritting_recognition_utils
python batch_wise_handwritting_image_ocr_processing.py --task_name repro_debug --API_model_name gemini-2.5-pro --split_name debug

# 8. Full reproduction (513 pages; check the cost first)
python batch_wise_handwritting_image_ocr_processing.py --task_name repro_obs --API_model_name gemini-2.5-pro --split_name observation
```

**[MANUAL]** Then go to `check_recognition_error_val\` and run the step 9 → 13 scripts in order (check the exact file names in that folder). Step 11 prints the **sample error rate** and **average error count**. Compare them with the paper.

---

## Part 3: GPU setup on Kaggle
**[MANUAL]**
- New Notebook → Settings → Accelerator: **GPU T4** · Internet: **On**.
- Add-ons → Secrets → add `HF_TOKEN`.
- To use the EDU data on Kaggle: upload the zip as a **private** Kaggle Dataset, then "Add Input" in the notebook.

**[TERMINAL]** notebook cells:
```python
!pip install -q -U transformers accelerate peft trl bitsandbytes qwen-vl-utils datasets
```
```python
import torch, os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="auto")
processor = AutoProcessor.from_pretrained(MODEL, min_pixels=256*28*28, max_pixels=1280*28*28)
print(torch.cuda.memory_allocated()/1e9, "GB used")   # expect ~8 GB
```
```python
# Smoke test: transcribe one FERMAT page
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
ds = load_dataset("ai4bharat/FERMAT", split="train")       # check the split name on the HF page
img = ds[0]["image"]
msgs = [{"role": "user", "content": [
    {"type": "image", "image": img},
    {"type": "text", "text": "Transcribe the student's solution exactly as written, including any mistakes. Do not correct anything."}]}]
text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
imgs, _ = process_vision_info(msgs)
inputs = processor(text=[text], images=imgs, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=512)
print(processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0])
print("GOLD (what the student wrote):", ds[0]["pert_a"])
```

**On Ada instead of Kaggle:** use the same `pip install` and Python code, minus the `kaggle_secrets` lines. Run `export HF_TOKEN=...` in the shell instead. If Ada has ≤12 GB VRAM, load in 4-bit by adding `quantization_config=BitsAndBytesConfig(load_in_4bit=True)`.

---

## Requirements at a glance
| Item | Need |
|---|---|
| Disk | ~25 GB (D: has 160 GB free) |
| Laptop RAM | 16 GB: fine for data and API work |
| GPU for 3B inference | ≥10 GB VRAM in fp16, or ≥6 GB in 4-bit |
| GPU for QLoRA fine-tuning | ≥14 GB VRAM (a T4 works; about 20 hours) |
| GPU machine RAM | ≥16 GB (Kaggle gives about 30 GB) |

## Done when
- [ ] The EDU debug split runs, and step 11 numbers are close to the paper
- [ ] FERMAT is downloaded on the laptop
- [ ] The Kaggle notebook prints a Qwen transcript next to the FERMAT `pert_a` gold text
