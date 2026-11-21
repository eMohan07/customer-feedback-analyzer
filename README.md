# Customer Feedback Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/vg4pj9hn84/badge.svg)](https://ask.thili.ai/b/vg4pj9hn84)

This Customer Feedback Analyzer helps retail managers quickly identify trends and prioritize improvements by automatically categorizing customer comments. It leverages a fine-tuned GPT-2 model to classify feedback, demonstrating the core fine-tuning and inference pipeline for text classification.

## What you are building

**Goal.** You will build a tool for retail managers that automatically categorizes customer feedback comments, such as 'product issue' or 'staff praise'.

**Done when.** You will have a fine-tuned GPT-2 model that can classify new customer comments by generating special label tokens. The project is complete when you can input a customer comment into a Streamlit app and receive its classification.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Define 3-5 feedback categories and create a small dataset of example comments with labels.**  
      Hint: Refer to the drill's approach for creating a labeled dataset for classification.
- [x] **2. Extend GPT-2 tokenizer and model with new category tokens.**  
      Hint: Follow the drill's steps for adding special tokens to the tokenizer and resizing the model's embedding layer.
- [ ] **3. Fine-tune the model using the Hugging Face Trainer.**  
      Hint: Utilize the Hugging Face `Trainer` class, as demonstrated in the drill, for fine-tuning.
- [ ] **4. Implement an inference function to classify new feedback.**  
      Hint: Adapt the drill's prediction logic to create a function that takes new text and returns a classification.
- [ ] **5. Build a simple Streamlit app to input comments and show classifications.**  
      Hint: Use `streamlit.text_area` for input and `streamlit.write` to display the classification.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Display the model's confidence score alongside the classified category.
- Allow users to input multiple comments at once and display all classifications.
- Add a 'clear input' button to the Streamlit app.
