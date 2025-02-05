import re
from datasets import load_dataset

# Load Wikipedia dataset
dataset = load_dataset("wikipedia", "20220301.simple", split="train")

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"[^a-zA-Z0-9.,!? ]", "", text)  # Remove special characters
    return text

dataset = dataset.map(lambda x: {"text": clean_text(x["text"])})
dataset.to_json("data/cleaned_wikipedia.json")
