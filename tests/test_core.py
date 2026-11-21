"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest
from app.core import (
    milestone_1,
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
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Fine-tune the model using the Hugging Face Trainer.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Implement an inference function to classify new feedback.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Build a simple Streamlit app to input comments and show classifications.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")
