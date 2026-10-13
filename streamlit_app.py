import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "Bianca255/kinnews-afroxlmr"

st.set_page_config(page_title="Kinyarwanda News Topic Classifier")


@st.cache_resource(show_spinner="Loading the model (the first load can take a few minutes)...")
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
    return tok, model


tok, model = load_model()

st.title("Kinyarwanda News Topic Classifier")
st.write(
    "AfroXLMR fine-tuned on KINNEWS (14 topics). "
    "It always picks one of the 14 topics, even for text outside the news domain."
)

text = st.text_area("Kinyarwanda news text", height=220)

if st.button("Classify"):
    # The training data was lowercased, so the input is lowercased too.
    clean = text.lower().strip()
    if not clean:
        st.warning("Please paste some Kinyarwanda text first.")
    else:
        enc = tok(clean, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)[0]
        scores = {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}
        top5 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
        st.subheader(f"Predicted topic: {top5[0][0]}")
        st.bar_chart(pd.DataFrame(top5, columns=["topic", "probability"]).set_index("topic"))
