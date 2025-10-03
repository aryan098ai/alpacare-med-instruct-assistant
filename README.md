# AlpaCare — Medical Instruction Assistant (LoRA / PEFT)

**Goal:** Fine-tune a pre-trained LLM (<7B) on the `lavita/AlpaCare-MedInstruct-52k` dataset to build a safe, **non-diagnostic** medical instruction assistant using LoRA/PEFT on Google Colab.  
**Note:** This assistant must never provide diagnosis, prescriptions, or dosage. Every output includes the disclaimer:

> *"This is educational only — consult a qualified clinician."*

---


## 📂 Repository Structure

📄 [README.md](README.md)  
📄 [requirements.txt](requirements.txt)  
📄 [data_loader.py](data_loader.py)  
📄 [REPORT.pdf](REPORT.pdf)  

📂 [human_eval](human_eval)  
├── 📄 [human_eval_rubric.csv](human_eval/human_eval_rubric.csv)  
└── 📄 [human_eval_responses.csv](human_eval/human_eval_responses.csv)  

📂 [adapters](adapters)  
└── [alpacare_lora_adapter.zip](alphacare_lora_adapter.zip) 

📂 [notebooks](notebooks)  
├── 📄 [colab-finetune.ipynb](notebooks/colab-finetune.ipynb)  
└── 📄 [inference_demo.ipynb](notebooks/inference_demo.ipynb)  

📄 [LICENSE](LICENSE)  



---

## 🚀 Quick start (Colab)
1. Open `notebooks/colab-finetune.ipynb` in Google Colab (GPU runtime).
2. Run all cells. If GPU memory is low, switch to the “subset mode” (small data slice).
3. Training saves LoRA adapter to `/content/drive/MyDrive/alpacare_adapter/` or `adapters/`.
4. Use `notebooks/inference_demo.ipynb` to load base model + adapter and test prompts (disclaimer included).

---

## 🧠 Model choice & license
- Suggested base models (pick one under 7B):
  - `stabilityai/stablelm-tuned-alpha-3b` (≈3B)  
  - `EleutherAI/gpt-neox-3.6b` (≈3.6B)  
- Confirm license terms before training.

---

## 📊 Dataset
- Source: `lavita/AlpaCare-MedInstruct-52k` (Hugging Face).
- Splits: 90% train, 5% val, 5% test.
- Subset used in demo: small slice for Colab GPU constraints.
- Preprocessing & cleaning in `data_loader.py`.

---

## 🛠️ Dependencies
See `requirements.txt`.  
Main libs: `transformers`, `datasets`, `accelerate`, `peft`, `bitsandbytes`, `torch`.

---

## 📦 Deliverables
- LoRA adapter (saved via `PeftModel.save_pretrained()`).
- `REPORT.pdf` (≤6 pages: dataset, approach, eval, safety).
- Human eval CSVs (≥30 reviews).
- Notebooks for training & inference.

---

## ⚠️ Safety
- Filtered training set: removed diagnosis/prescription samples.
- All outputs include disclaimer.
- Human evaluation by medically literate reviewers.
- Not for clinical use.

---

## 👤 Maintainer
- Aryan Shukla / aryan098ai / shuklaaryan070@gmail.com
