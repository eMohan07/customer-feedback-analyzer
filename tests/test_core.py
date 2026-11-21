"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import os
from unittest.mock import patch
import pytest
import torch
from transformers import Trainer
from app.core import (
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    classify_feedback,
    classify_feedback_with_confidence,
    get_categories,
    get_label_tokens,
    format_sample,
)

# Milestone 1: Define 3-5 feedback categories and create a small dataset of example comments with labels.
def test_milestone_1():
    dataset = milestone_1()
    categories = get_categories()
    label_tokens = get_label_tokens()

    # Must define between 3 and 5 categories
    assert 3 <= len(categories) <= 5
    assert "product issue" in categories
    assert "staff praise" in categories

    # Each category must map to a unique special label token
    assert len(label_tokens) == len(categories)
    for cat in categories:
        assert cat in label_tokens
        token = label_tokens[cat]
        assert token.startswith("[") and token.endswith("]")

    # Dataset should contain examples for all categories
    assert len(dataset) >= 15
    seen_categories = set()
    for row in dataset:
        assert row["label"] in categories
        assert len(row["text"].strip()) > 0
        assert row["label_token"] == label_tokens[row["label"]]
        assert "formatted_text" in row
        assert row["text"] in row["formatted_text"]
        assert row["label_token"] in row["formatted_text"]
        seen_categories.add(row["label"])

    assert seen_categories == set(categories)


# Milestone 2: Extend GPT-2 tokenizer and model with new category tokens.
def test_milestone_2():
    tokenizer, model = milestone_2()
    label_tokens = get_label_tokens()

    # Tokenizer has pad token
    assert tokenizer.pad_token is not None

    # Check that all special category tokens are recognized by the tokenizer
    for token in label_tokens.values():
        token_id = tokenizer.convert_tokens_to_ids(token)
        assert token_id is not None
        assert token_id != tokenizer.unk_token_id
        # Ensure roundtrip decoding works
        decoded = tokenizer.decode([token_id])
        assert token in decoded

    # Model embedding layer has been resized to match tokenizer vocab size
    assert model.get_input_embeddings().weight.shape[0] == len(tokenizer)


# Milestone 3: Fine-tune the model using the Hugging Face Trainer.
def test_milestone_3():
    # Run fine-tuning with 1 step to verify the Hugging Face Trainer pipeline executes properly
    result = milestone_3(max_steps=1)
    trainer, model, tokenizer = result

    assert isinstance(trainer, Trainer)
    assert trainer.state.global_step >= 1
    assert model is not None
    assert tokenizer is not None


# Milestone 4: Implement an inference function to classify new feedback.
def test_milestone_4():
    tokenizer, model = milestone_2()
    inference_fn = milestone_4()
    categories = get_categories()

    test_comment = "The cashier at checkout was very polite and helpful."
    prediction = inference_fn(test_comment, model=model, tokenizer=tokenizer)
    assert prediction in categories

    # Also test direct milestone_4 execution with text
    prediction_direct = milestone_4(test_comment, model=model, tokenizer=tokenizer)
    assert prediction_direct in categories

    # Verify classify_feedback_with_confidence returns category and valid probability
    cat, conf, scores = classify_feedback_with_confidence(test_comment, model=model, tokenizer=tokenizer)
    assert cat in categories
    assert 0.0 <= conf <= 1.0
    assert len(scores) == len(categories)


# Milestone 5: Build a simple Streamlit app to input comments and show classifications.
def test_milestone_5():
    app_path = milestone_5()
    assert os.path.exists(app_path)

    # Verify app/main.py uses text_area and write as specified
    with open(app_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "st.text_area" in content
    assert "st.write" in content

    # Test Streamlit UI rendering and interactivity using AppTest
    from streamlit.testing.v1 import AppTest

    class DummyTokenizer:
        pad_token_id = 0
        eos_token_id = 0
        def convert_tokens_to_ids(self, tok):
            return 1
        def decode(self, *args, **kwargs):
            return "[STAFF_PRAISE]"
        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    class DummyModel:
        device = "cpu"
        def __call__(self, *args, **kwargs):
            class DummyOutput:
                logits = torch.randn(1, 3, 50261)
            return DummyOutput()
        def generate(self, *args, **kwargs):
            return torch.tensor([[1, 2, 3, 4]])

    with patch("app.main.get_model", return_value=(DummyTokenizer(), DummyModel())):
        at = AppTest.from_file(app_path).run(timeout=10)
        assert not at.exception

        # Input feedback in text_area and trigger classify
        at.text_area[0].input("The store staff were very welcoming.").run(timeout=10)
        at.button[0].click().run(timeout=10)
        assert not at.exception
