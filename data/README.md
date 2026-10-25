# Data

Datasets are NOT stored in git. Download them as described below.

| Dataset | Use | Where to get it | Put it in |
|---|---|---|---|
| EDU-CIRCUIT-HW (Findings of ACL 2026) | Main evaluation: 1,334 real university circuit-analysis answers. 513 have expert verbatim transcripts and grades. | https://doi.org/10.35090/gatech/81252 (manual download) | unzip into `external/CIRCUIT_EDU_HW_ACL/` (folders: `Screenshot_output_anon`, `Rectified_recognized_markdown_done_Anon`, `Observationset_Final`, ...) |
| FERMAT (ACL 2025) | Fine-tuning the faithful reader + measuring silent corrections (planted errors: `pert_a` = as written, `orig_a` = correct) | https://huggingface.co/datasets/ai4bharat/FERMAT (gated: accept the terms) | Hugging Face cache, e.g. `data/hf_cache/` |

Splits (fixed, as pre-registered in the proposal):
* EDU observation set: split by student. 3 students for development, 8 for evaluation.
* FERMAT: split by problem (`orig_q`). About 1,650 train / 200 dev / 400 test.
