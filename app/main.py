"""Streamlit entry point: `streamlit run app/main.py`"""
import streamlit as st
from app.core import (
    milestone_2,
    classify_feedback_with_confidence,
    get_categories,
    RAW_FEEDBACK_DATA,
)

st.set_page_config(page_title="Customer Feedback Analyzer", page_icon="🛍️", layout="centered")

@st.cache_resource
def get_model():
    """Loads and caches the fine-tuned GPT-2 tokenizer and model."""
    return milestone_2()

st.title("Customer Feedback Analyzer")
st.write(
    "Retail managers use this tool to automatically categorize customer feedback comments "
    '(e.g., "product issue", "staff praise") to quickly identify trends and prioritize improvements.'
)

# Display supported categories
categories = get_categories()
st.info(f"**Supported Categories:** {', '.join(categories)}")

# Example feedback selection
example_options = ["(Custom Input)"] + [f"{item['label'].title()}: \"{item['text']}\"" for item in RAW_FEEDBACK_DATA[:6]]
selected_example = st.selectbox("Choose an example or enter your own below:", example_options)

default_text = ""
if selected_example != "(Custom Input)":
    # Extract the quote text
    default_text = selected_example.split(": \"", 1)[1].rstrip("\"")

# Input text area for comments (as required by Milestone 5)
comment = st.text_area(
    "Customer feedback comment:",
    value=default_text,
    height=120,
    placeholder="e.g., The cashier was super friendly and helped pack my items.",
)

col1, col2 = st.columns([1, 4])
with col1:
    classify_clicked = st.button("Classify", type="primary")
with col2:
    clear_clicked = st.button("Clear")

if clear_clicked:
    st.rerun()

if classify_clicked or (comment and default_text):
    if comment and comment.strip():
        with st.spinner("Classifying feedback..."):
            try:
                tokenizer, model = get_model()
                category, confidence, scores = classify_feedback_with_confidence(
                    comment.strip(), model=model, tokenizer=tokenizer
                )
                st.subheader("Classification Result")
                # Using streamlit.write as specified in milestone 5
                st.write(f"**Category:** {category}")
                st.write(f"**Confidence:** {confidence:.1%}")

                with st.expander("Category Probabilities"):
                    for cat, prob in scores.items():
                        st.write(f"- **{cat}**: {prob:.1%}")
            except Exception as e:
                st.error(f"Error classifying feedback: {e}")
    else:
        st.warning("Please enter a customer feedback comment above to classify.")
