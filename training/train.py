import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.transformer import MiniGPT

# Load dataset
dataset = load_dataset("wikipedia", "20220301.simple", split="train")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.remove_columns(["text"])

# Convert dataset into PyTorch format
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return torch.tensor(self.dataset[idx]["input_ids"])

train_dataloader = DataLoader(TextDataset(dataset), batch_size=16, shuffle=True)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniGPT(tokenizer.vocab_size).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-4)
loss_fn = nn.CrossEntropyLoss()

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        inputs = batch.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs.view(-1, tokenizer.vocab_size), inputs.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_dataloader)}")

# Save model
torch.save(model.state_dict(), "training/minigpt.pth")
