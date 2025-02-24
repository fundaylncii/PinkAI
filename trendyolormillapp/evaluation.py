
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(model_path):
    """Kayıtlı modeli ve tokenizer'ı yükler."""
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def get_sentiment(text, model, tokenizer):
    """Verilen metnin sentiment tahminini döndürür."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return "positive" if predicted_class == 1 else "negative"

def evaluate_model(df, model, tokenizer):
    """DataFrame'deki yorumlar için sentiment analizi yapar ve sonuçları döndürür."""
    df["sentiment"] = df["comment"].apply(lambda x: get_sentiment(x, model, tokenizer))
    return df
