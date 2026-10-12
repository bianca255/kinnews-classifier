import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "Bianca255/kinnews-afroxlmr"

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device).eval()


def classify(text):
    # The training data was lowercased, so the input is lowercased too.
    text = (text or "").lower().strip()
    if not text:
        return {}
    enc = tok(text, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0].cpu()
    return {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=8, label="Kinyarwanda news text"),
    outputs=gr.Label(num_top_classes=5, label="Predicted topic"),
    title="Kinyarwanda News Topic Classifier",
    description=(
        "AfroXLMR fine-tuned on KINNEWS (14 topics). "
        "It always picks one of the 14 topics, even for text outside the news domain."
    ),
)

if __name__ == "__main__":
    demo.launch()  # use demo.launch(share=True) in Colab for a temporary public link
