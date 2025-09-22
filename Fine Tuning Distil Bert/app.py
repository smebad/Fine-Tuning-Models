import gradio as gr
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load finetuned model & tokenizer
model_path = "./distilbert-finetuned"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# label mapping from model config
id2label = model.config.id2label

# Prediction function
def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return id2label[predicted_class]

# Gradio UI
iface = gr.Interface(
    fn=predict_intent,
    inputs=gr.Textbox(lines=2, placeholder="Enter your sentence..."),
    outputs="text",
    title="DistilBERT Intent Classifier",
    description="Enter a sentence and get the predicted intent."
)

if __name__ == "__main__":
    iface.launch(share=True)
