"""Your code. One stub per milestone; build them in order and keep each one small."""
import os
import tempfile
import torch
from datasets import Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

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

class TrainingResult(tuple):
    """Container holding trainer, model, and tokenizer, supporting tuple unpacking and attribute access."""
    def __new__(cls, trainer, model, tokenizer):
        return super(TrainingResult, cls).__new__(cls, (trainer, model, tokenizer))

    def __init__(self, trainer, model, tokenizer):
        self.trainer = trainer
        self.model = model
        self.tokenizer = tokenizer


def milestone_3(
    model=None,
    tokenizer=None,
    dataset=None,
    output_dir: str = None,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_steps: int = None,
):
    """
    Fine-tunes the model on the feedback classification dataset using Hugging Face Trainer.
    """
    if tokenizer is None or model is None:
        tok, mod = milestone_2()
        tokenizer = tokenizer or tok
        model = model or mod

    if dataset is None:
        dataset = create_feedback_dataset()

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="gpt2_feedback_")

    # Format texts with EOS token for causal language model training
    texts_with_eos = [item + tokenizer.eos_token for item in dataset["formatted_text"]]
    tokenized = tokenizer(
        texts_with_eos,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors=None,
    )
    train_dataset = Dataset.from_dict(tokenized)

    training_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": per_device_train_batch_size,
        "learning_rate": learning_rate,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "use_cpu": not torch.cuda.is_available(),
    }
    if max_steps is not None:
        training_kwargs["max_steps"] = max_steps
    else:
        training_kwargs["num_train_epochs"] = num_train_epochs

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    return TrainingResult(trainer, model, tokenizer)


# --- Milestone 4: Implement an inference function to classify new feedback.
# Hint: Adapt the drill's prediction logic to create a function that takes new text and returns a classification.

def classify_feedback(text: str, model=None, tokenizer=None) -> str:
    """
    Classifies a customer feedback comment into one of the defined categories.
    Uses GPT-2 model and tokenizer with special category label tokens.
    """
    if tokenizer is None or model is None:
        tok, mod = milestone_2()
        tokenizer = tokenizer or tok
        model = model or mod

    prompt = f"Feedback: {text}\nCategory:"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]

    # Evaluate logits for each candidate category token
    cat_scores = {}
    for cat, token_str in LABEL_TOKENS.items():
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        cat_scores[cat] = next_token_logits[token_id].item()

    best_category = max(cat_scores, key=cat_scores.get)

    # Attempt generation to check for exact label token output
    try:
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        new_ids = gen_outputs[0][inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(new_ids, skip_special_tokens=False).strip()
        for token_str, cat in TOKEN_TO_LABEL.items():
            if token_str in gen_text:
                return cat
    except Exception:
        pass

    return best_category


def classify_feedback_with_confidence(text: str, model=None, tokenizer=None):
    """
    Classifies a customer feedback comment and returns category, confidence, and scores dict.
    """
    if tokenizer is None or model is None:
        tok, mod = milestone_2()
        tokenizer = tokenizer or tok
        model = model or mod

    prompt = f"Feedback: {text}\nCategory:"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]

    token_ids = [tokenizer.convert_tokens_to_ids(LABEL_TOKENS[c]) for c in CATEGORIES]
    logits_subset = torch.stack([next_token_logits[tid] for tid in token_ids])
    probs = torch.softmax(logits_subset, dim=-1)

    scores = {cat: probs[i].item() for i, cat in enumerate(CATEGORIES)}
    best_cat = max(scores, key=scores.get)
    confidence = scores[best_cat]
    return best_cat, confidence, scores


def milestone_4(text: str = None, model=None, tokenizer=None):
    """
    Milestone 4: Implement an inference function to classify new feedback.
    Returns the inference function or executes classification if parameters are provided.
    """
    if text is not None:
        return classify_feedback(text, model=model, tokenizer=tokenizer)
    return classify_feedback


# --- Milestone 5: Build a simple Streamlit app to input comments and show classifications.
# Hint: Use `streamlit.text_area` for input and `streamlit.write` to display the classification.
def milestone_5():
    """
    Milestone 5: Build a simple Streamlit app to input comments and show classifications.
    Returns the absolute path to the Streamlit app script.
    """
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "main.py"))
    return app_path
