import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from train import MiniGPT

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and load trained weights
model = MiniGPT(tokenizer.vocab_size).to(device)
model.load_state_dict(torch.load("minigpt.pth", map_location=device))
model.eval()

def generate_text(prompt, max_length=50):
    """Generate text from the trained MiniGPT model."""
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token = torch.argmax(F.softmax(outputs[:, -1, :], dim=-1), dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    return tokenizer.decode(input_ids[0])

# Example usage
if __name__ == "__main__":
    prompt = input("Enter a prompt: ")
    print(generate_text(prompt))
