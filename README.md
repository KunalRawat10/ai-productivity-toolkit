# 🚀 AI Productivity Toolkit

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![Gradio](https://img.shields.io/badge/UI-Gradio%20App-orange)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A production-ready AI workflow suite powered by open-weights Small Language Models (SLMs). This toolkit demonstrates role conditioning, schema-enforced JSON extraction, code debugging, and interactive browser deployment without relying on closed paid APIs.

---

## 📌 Features

* **🧠 Executive Technical Summarizer:** Condenses lengthy papers, documentation, and technical articles into 3 high-signal bullet points and a single key takeaway.
* **🏷️ Structured JSON Extractor:** Converts raw, unstructured text (resumes, bios, logs) directly into validated JSON schemas for database ingestion.
* **🛠️ Automated Code Debugger & Refactorer:** Diagnoses logic errors, off-by-one bounds, and anti-patterns in Python snippets and provides instant refactored solutions.
* **📬 Value-First Outreach Generator:** Drafts targeted, high-conversion cold emails under 100 words for professional networking and research inquiries.
* **🌐 Interactive Gradio Web Interface:** Includes a built-in web UI deployable locally or shareable via temporary public tunnels.

---

## 🏗️ System Architecture


```text
User Input / Web UI (Gradio)
        │
        ▼
ChatML Prompt Template Engine (Roles, Contexts, Constraints)
        │
        ▼
Hugging Face Pipeline (SmolLM2-135M-Instruct)
        ├── Temperature Calibration (T = 0.2)
        ├── Nucleus Sampling (Top-p = 0.9)
        └── BPE Tokenizer Decoding
        │
        ▼
Clean Structured Output (JSON / Formatted Text / Code)
```

---

## ⚙️ Installation & Setup

### Option 1: Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/KunalRawat10/ai-productivity-toolkit.git](https://github.com/KunalRawat10/ai-productivity-toolkit.git)
   cd ai-productivity-toolkit
   
2. Create a virtual environment & install dependencies:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch transformers accelerate gradio


3. Run the application:
4. 
python toolkit.py

Option 2: Google Colab (Zero Installation)
Run the interactive notebook directly in your browser:

from toolkit import AIProductivityToolkit

# Initialize toolkit
toolkit = AIProductivityToolkit()

# 1. Executive Summarization
summary = toolkit.summarize_executive("Your long technical paper text here...")
print(summary)

# 2. Structured JSON Entity Extraction
raw_bio = "Kunal Rawat is an AIML engineer with 2 years of experience in PyTorch and LangChain."
entities = toolkit.extract_structured_entities(raw_bio)
print(entities)

# 3. Code Refactoring
buggy_code = "def add_item(val, lst=[]): lst.append(val); return lst"
fixed_code = toolkit.code_debugger_refactor(buggy_code, "Default mutable argument bug")
print(fixed_code)

👨‍💻 Author
Kunal Rawat
GitHub: @KunalRawat10
LinkedIn: Kunal Rawat

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
