# 🛍️ Customer Feedback Analyzer

[![CI Tests](https://github.com/eMohan07/customer-feedback-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/eMohan07/customer-feedback-analyzer/actions)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange)](https://huggingface.co/docs/transformers/index)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.49-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)

An automated feedback classification system designed for retail managers to quickly detect customer sentiment trends, operational bottlenecks, and service highlights. The tool leverages a **fine-tuned GPT-2 generative language model** extended with custom category tokens, allowing it to classify raw customer feedback into domain-specific categories with probability confidence scores via an interactive **Streamlit** dashboard.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture & Classification Approach](#-architecture--classification-approach)
- [Feedback Categories](#-feedback-categories)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Streamlit Web App](#running-the-streamlit-web-app)
  - [Running Tests](#running-tests)
- [Pipeline Breakdown](#-pipeline-breakdown)
  - [1. Dataset & Prompt Structuring](#1-dataset--prompt-structuring)
  - [2. Tokenizer & Embedding Resizing](#2-tokenizer--embedding-resizing)
  - [3. Fine-Tuning with Trainer](#3-fine-tuning-with-trainer)
  - [4. Generative Next-Token Inference](#4-generative-next-token-inference)
  - [5. Streamlit Interactive Dashboard](#5-streamlit-interactive-dashboard)
- [Milestones Progress](#-milestones-progress)
- [License](#-license)

---

## 🚀 Overview

Retail managers deal with high volumes of incoming customer feedback across multiple touchpoints (in-store checkouts, online reviews, receipts, and customer service desks). Manually sorting these comments is time-consuming and prone to inconsistencies.

The **Customer Feedback Analyzer** automates this classification pipeline by:
1. Training a GPT-2 model to recognize and generate dedicated classification tokens directly following a structured prompt format.
2. Providing a clean, intuitive Streamlit web UI where managers can test customer comments, view category classifications, and inspect confidence probabilities.

---

## 🧠 Architecture & Classification Approach

Instead of using a traditional external classification head, this system implements **Generative Token Classification**:

```
Raw Customer Comment
        │
        ▼
Prompt Formatting: "Feedback: {text}\nCategory:"
        │
        ▼
Extended GPT-2 Model (50,261 vocab with added label tokens)
        │
        ▼
Next-Token Prediction over Special Tokens ([PRODUCT_ISSUE], [STAFF_PRAISE], etc.)
        │
        ▼
Softmax Confidence & Generated Special Token Decoding
        │
        ▼
Classification Result & Streamlit Visualization
```

---

## 🏷️ Feedback Categories

The model is configured with 4 core retail feedback categories, each represented by a dedicated special token:

| Category | Special Label Token | Description / Examples |
| :--- | :--- | :--- |
| **Product Issue** | `[PRODUCT_ISSUE]` | Damaged goods, defective items, broken packaging, leaks. |
| **Staff Praise** | `[STAFF_PRAISE]` | Friendly cashiers, polite associates, helpful floor managers. |
| **Store Cleanliness** | `[STORE_CLEANLINESS]` | Restroom conditions, aisle spills, neatness, organized shelves. |
| **Billing Problem** | `[BILLING_PROBLEM]` | Overcharging, coupon failures, register discrepancies, tax errors. |

---

## ✨ Key Features

- **Custom Special Tokens**: Extends GPT-2's vocabulary with dedicated domain tokens and automatically resizes model embeddings to preserve weight alignment.
- **Hugging Face `Trainer` Integration**: Native integration with the Hugging Face `Trainer` and `DataCollatorForLanguageModeling` for standard fine-tuning workflows.
- **Dual Inference Engine**: Combines greedy token generation with softmax probability scoring over label tokens for high-speed, reliable predictions and confidence estimation.
- **Interactive Streamlit App**: User-friendly UI featuring `streamlit.text_area` input, `streamlit.write` output, instant example selection, and confidence distributions.
- **Automated CI/CD**: Fully tested with pytest (unit tests for plumbing, core pipelines, and Streamlit `AppTest` integration).

---

## 📂 Repository Structure

```text
customer-feedback-analyzer/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI workflow (Python 3.12 test matrix)
├── app/
│   ├── __init__.py
│   ├── core.py                # Core ML pipeline (Milestones 1-5 implementations)
│   ├── llm.py                 # LLM configuration helper
│   ├── main.py                # Streamlit user interface entry point
│   └── storage.py             # Local storage persistence utility
├── tests/
│   ├── test_core.py           # Unit tests for Milestones 1 through 5
│   └── test_smoke.py          # Smoke tests for environment, storage, and compilation
├── .env.example               # Example environment variable file
├── .gitignore                 # Git ignore configuration
├── Dockerfile                 # Docker containerization config
├── pytest.ini                 # Pytest configuration settings
├── requirements.txt           # Python package dependencies
└── README.md                  # Project documentation
```

---

## 🛠️ Getting Started

### Prerequisites

- **Python 3.12+**
- **Git**
- Optional: CUDA-compatible GPU (the pipeline automatically runs on CPU if no GPU is available).

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/eMohan07/customer-feedback-analyzer.git
   cd customer-feedback-analyzer
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On macOS/Linux:
   python3 -m venv .venv
   source .venv/bin/activate

   # On Windows:
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```

---

### Running the Streamlit Web App

Start the Streamlit dashboard locally:

```bash
streamlit run app/main.py
```

Once running, open your browser at `http://localhost:8501`. You can:
- Select from preloaded retail feedback examples.
- Type or paste custom customer comments into the text area.
- Click **Classify** to see the predicted category and confidence rating.

---

### Running Tests

Execute the automated test suite with pytest:

```bash
pytest -v
```

All 8 tests across `test_smoke.py` and `test_core.py` will execute and report status.

---

## 🔬 Pipeline Breakdown

### 1. Dataset & Prompt Structuring
The dataset pairs customer comments with categories, formatted into prompts:
```text
Feedback: Sarah at the checkout was incredibly polite and helped pack my bags.
Category: [STAFF_PRAISE]
```

### 2. Tokenizer & Embedding Resizing
- Adds `tokenizer.pad_token = tokenizer.eos_token` to handle batch padding.
- Registers category labels as `additional_special_tokens`.
- Calls `model.resize_token_embeddings(len(tokenizer))` to expand the embedding layer from 50,257 to 50,261.

### 3. Fine-Tuning with Trainer
Utilizes Hugging Face's `Trainer` with `DataCollatorForLanguageModeling(mlm=False)` to train causal language modeling loss over feedback sequences.

### 4. Generative Next-Token Inference
Prompts the model with:
```text
Feedback: <New Customer Feedback>
Category:
```
Evaluates the logits of the next token across the special token IDs, computes softmax probabilities, and returns the top predicted category.

### 5. Streamlit Interactive Dashboard
Built with `streamlit.text_area` for comment entry and `streamlit.write` for displaying the classification and confidence level.

---

## 📋 Milestones Progress

- [x] **1. Define 3-5 feedback categories and create a small dataset of example comments with labels.**  
      *Implemented 4 categories (`product issue`, `staff praise`, `store cleanliness`, `billing problem`) and an example dataset.*
- [x] **2. Extend GPT-2 tokenizer and model with new category tokens.**  
      *Added special tokens, configured padding, and resized GPT-2 embedding layer.*
- [x] **3. Fine-tune the model using the Hugging Face Trainer.**  
      *Implemented causal LM fine-tuning using Hugging Face `Trainer`.*
- [x] **4. Implement an inference function to classify new feedback.**  
      *Created `classify_feedback` and `classify_feedback_with_confidence` with next-token decoding.*
- [x] **5. Build a simple Streamlit app to input comments and show classifications.**  
      *Constructed interactive Streamlit interface in `app/main.py` with `st.text_area` and `st.write`.*

---

## 📄 License

This project is licensed under the Apache 2.0 License.
