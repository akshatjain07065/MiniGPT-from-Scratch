import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ============================
# 1️⃣ Load and Preprocess Dataset
# ============================

print("Loading dataset...")
dataset = load_dataset("wikipedia", "20220301.simple", split="train")

# Use a pre-trained tokenizer (GPT-2)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

# Apply tokenization
dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.remove_columns(["text"])  # Keep only input_ids

# Convert dataset into PyTorch format
class TextDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return torch.tensor(item["input_ids"]), torch.tensor(item["input_ids"])  # Input & Target

train_dataloader = DataLoader(TextDataset(dataset), batch_size=16, shuffle=True)

# ============================
# 2️⃣ Define Transformer Model
# ============================

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.attention(x, x, x)[0]
        x = self.norm1(x + attention)
        forward = self.feed_forward(x)
        x = self.norm2(x + forward)
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_size=256, num_layers=4, heads=8, dropout=0.1, forward_expansion=4):
        super(MiniGPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        logits = self.fc_out(x)
        return logits

# ============================
# 3️⃣ Training Setup
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MiniGPT(tokenizer.vocab_size).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-4)
loss_fn = nn.CrossEntropyLoss()

# ============================
# 4️⃣ Training Loop
# ============================

epochs = 3
print("Starting training...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch in loop:
        optimizer.zero_grad()
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs.view(-1, tokenizer.vocab_size), targets.view(-1))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1}: Avg Loss = {total_loss / len(train_dataloader)}")

# ============================
# 5️⃣ Save Model
# ============================

torch.save(model.state_dict(), "minigpt.pth")
print("Model saved as minigpt.pth")
