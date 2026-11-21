"""Your code. One stub per milestone; build them in order and keep each one small."""
import os
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# --- Milestone 1: Define 3-5 feedback categories and create a small dataset of example comments with labels.
# Hint: Refer to the drill's approach for creating a labeled dataset for classification.

CATEGORIES = [
    "product issue",
    "staff praise",
    "store cleanliness",
    "billing problem",
]

LABEL_TOKENS = {
    "product issue": "[PRODUCT_ISSUE]",
    "staff praise": "[STAFF_PRAISE]",
    "store cleanliness": "[STORE_CLEANLINESS]",
    "billing problem": "[BILLING_PROBLEM]",
}

TOKEN_TO_LABEL = {v: k for k, v in LABEL_TOKENS.items()}

RAW_FEEDBACK_DATA = [
    # Product issue
    {"text": "The blender stopped working after two days of use.", "label": "product issue"},
    {"text": "The zipper on this jacket broke the very first time I wore it.", "label": "product issue"},
    {"text": "The milk carton was leaking all over my shopping bag.", "label": "product issue"},
    {"text": "The screen on this smartwatch has dead pixels right out of the box.", "label": "product issue"},
    {"text": "These shoes started peeling at the seams within a week.", "label": "product issue"},
    {"text": "The packaging was crushed and the ceramic mug arrived broken.", "label": "product issue"},
    # Staff praise
    {"text": "Sarah at the checkout was incredibly polite and helped pack my bags.", "label": "staff praise"},
    {"text": "The customer service representative went above and beyond to locate my item.", "label": "staff praise"},
    {"text": "The cashier was very friendly and made my shopping experience wonderful.", "label": "staff praise"},
    {"text": "Great assistance from David in the electronics department, very knowledgeable.", "label": "staff praise"},
    {"text": "The manager was patient and solved my return issue with a warm smile.", "label": "staff praise"},
    {"text": "The store associate gladly guided me to the aisle I was looking for.", "label": "staff praise"},
    # Store cleanliness
    {"text": "The aisles in the produce section were messy and had spilled fruit.", "label": "store cleanliness"},
    {"text": "Restrooms on the second floor were out of soap and very dirty.", "label": "store cleanliness"},
    {"text": "The store floors were sparkling clean and the shelves were well organized.", "label": "store cleanliness"},
    {"text": "Spill in aisle 4 had no warning sign and trash was overflowing.", "label": "store cleanliness"},
    {"text": "The shopping carts were spotless and the entrance was very neat.", "label": "store cleanliness"},
    {"text": "Dust everywhere on the display shelves and litter near checkout.", "label": "store cleanliness"},
    # Billing problem
    {"text": "I was charged twice for the same box of cereal on my receipt.", "label": "billing problem"},
    {"text": "The discount coupon was not applied to my total bill at checkout.", "label": "billing problem"},
    {"text": "Overcharged on the promotional item that was marked 50% off on the shelf.", "label": "billing problem"},
    {"text": "The register showed a higher price than the price tag displayed.", "label": "billing problem"},
    {"text": "My gift card balance was deducted but the payment failed to go through.", "label": "billing problem"},
    {"text": "An incorrect sales tax rate was added to my final invoice.", "label": "billing problem"},
]


def format_sample(text: str, label: str) -> str:
    """Formats feedback comment and its label token into a prompt string."""
    token = LABEL_TOKENS.get(label, label)
    return f"Feedback: {text}\nCategory: {token}"


def get_categories():
    """Returns the list of defined feedback categories."""
    return list(CATEGORIES)


def get_label_tokens():
    """Returns the mapping from categories to special tokens."""
    return dict(LABEL_TOKENS)


def create_feedback_dataset() -> Dataset:
    """Creates a Hugging Face Dataset from example comments and labels."""
    records = []
    for item in RAW_FEEDBACK_DATA:
        records.append({
            "text": item["text"],
            "label": item["label"],
            "label_token": LABEL_TOKENS[item["label"]],
            "formatted_text": format_sample(item["text"], item["label"]),
        })
    return Dataset.from_list(records)


def milestone_1():  # Milestone 1
    return create_feedback_dataset()


# --- Milestone 2: Extend GPT-2 tokenizer and model with new category tokens.
# Hint: Follow the drill's steps for adding special tokens to the tokenizer and resizing the model's embedding layer.
def milestone_2(model_name: str = "gpt2"):
    """
    Extends GPT-2 tokenizer and model with new category tokens.
    Adds pad token if missing, registers additional special tokens,
    and resizes the model's embedding layer to match tokenizer length.
    """
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokens_to_add = [tok for tok in LABEL_TOKENS.values() if tok not in tokenizer.get_vocab()]
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

    try:
        model = GPT2LMHeadModel.from_pretrained(model_name, local_files_only=True)
    except Exception:
        model = GPT2LMHeadModel.from_pretrained(model_name)

    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


# --- Milestone 3: Fine-tune the model using the Hugging Face Trainer.
# Hint: Utilize the Hugging Face `Trainer` class, as demonstrated in the drill, for fine-tuning.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Implement an inference function to classify new feedback.
# Hint: Adapt the drill's prediction logic to create a function that takes new text and returns a classification.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Build a simple Streamlit app to input comments and show classifications.
# Hint: Use `streamlit.text_area` for input and `streamlit.write` to display the classification.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
