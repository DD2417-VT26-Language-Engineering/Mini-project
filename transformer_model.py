import os
import json

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.data import DataLoader, Dataset, random_split
from torch import nn
from torch.optim import Adam
import torch
from self_attention import MultiHeadSelfAttention


class TextDataset(Dataset):
    def __init__(self, token_ids, context_len=64):
        self.data = token_ids
        self.ctx = context_len

    def __len__(self):
        return len(self.data) - self.ctx

    def __getitem__(self, i):
        x = self.data[i : i + self.ctx]
        y = self.data[i + 1 : i + self.ctx + 1]  # shifted by 1
        return torch.tensor(x), torch.tensor(y)
    
from dataclasses import dataclass

@dataclass
class Config :
    vocab_size: int = 5000  # This number should agree with the tokenizer
    number_of_transformer_blocks: int = 4
    number_of_attention_heads: int = 4
    vector_dim: int = 256
    block_size: int = 512
    dropout_prob: float = 0.1
    batch_size: int = 8
    learning_rate: float = 0.0005
    weight_decay: float = 0.000001
    no_of_epochs: int = 1

class PositionwiseFFN(nn.Module):
    """
    The position-wise FFN that follows after the self-attention computation.
    Vectors are projected to 4x the dimensionality and then projected down
    again after relu application.
    """

    def __init__(self, vector_dim, dropout_prob) :
        super().__init__()
        self.fc1 = nn.Linear(vector_dim, 4*vector_dim, bias=True)
        self.fc2 = nn.Linear(4*vector_dim, vector_dim, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))

class Block(nn.Module):
    """
    Transformer encoder block.

    This version differs from the original version in  [Vaswani et al. NeurIPS 2017],
    and applies the LayerNorm before the self-attention, and before the FFN, as this
    has proved to be beneficial (see [Nguyen and Salazar 2019]).
    """

    def __init__(self, vector_dim, n_heads, block_size, dropout_prob):
        super().__init__()
        att_dim = vector_dim // n_heads
        self.attn = MultiHeadSelfAttention(vector_dim, n_heads, block_size, is_causal=True)
        self.ffn = PositionwiseFFN(vector_dim, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.ln1 = nn.LayerNorm(vector_dim)
        self.ln2 = nn.LayerNorm(vector_dim)

    def forward(self, x):
        x1 = self.ln1(x)
        x2 = x + self.dropout(self.attn(x1))
        x3 = self.ln2(x2)
        x4 = x2 + self.dropout(self.ffn(x3))
        return x4

class TinyStoriesLM(nn.Module):

    def __init__(self, config):
        super(TinyStoriesLM, self).__init__()
        self.config = config
        self.embed =  nn.Embedding(config.vocab_size, config.vector_dim)
        self.positional = nn.Parameter(torch.randn(1, config.block_size, config.vector_dim))
        modules = [Block(config.vector_dim,\
                         config.number_of_attention_heads,\
                         config.block_size,\
                         config.dropout_prob) for _ in range(config.number_of_transformer_blocks)]
        self.transformers = nn.ModuleList(modules)
        self.final = nn.Linear(config.vector_dim, config.vocab_size)

    def forward(self, x):

        # YOUR CODE HERE
        B, T = x.shape
        token_embed = self.embed(x) # (B, T, C)
        pos_embed = self.positional[:, :T, :] # (1, T, C)
        x = token_embed + pos_embed # (B, T, C)

        for block in self.transformers:
            x = block(x)

        logits = self.final(x)
        
        
        return logits
    

def tokenize_and_save(tokenizer, filename, save_filename):
    with open(filename, "r") as f:
        texts = f.read().splitlines()
    encoded_batch = tokenizer.encode_batch(texts)
    print("Batch encoding complete")
    token_ids = []
    for i, encoded in enumerate(encoded_batch):
        token_ids.extend(encoded.ids)
    print(f"Total tokens: {len(token_ids)}")
    print("Saving token IDs...")
    with open(save_filename, "w") as f:
        json.dump(token_ids, f)

    return token_ids

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train(model, dataloader, val_dataloader, epochs=1, lr=1e-3, verbose=True):
    device = next(model.parameters()).device
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)  # (B, T, vocab_size)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if verbose and (i + 1) % 100 == 0:
                avg_loss = total_loss / 100
                print(f"Epoch {epoch+1}, Step {i+1}/{len(dataloader)}, Loss: {avg_loss:.4f}")
                total_loss = 0
            if verbose and (i + 1) % 1000 == 0:
                val_loss = evaluate(model, val_dataloader, criterion, device)
                model.train()
                print(f"Epoch {epoch+1}, Step {i+1}/{len(dataloader)}, Validation Loss: {val_loss:.4f}")

def get_tokenizer():

    if os.path.exists("tokenizer.json"):
        return Tokenizer.from_file("tokenizer.json")

    # Create tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=5000,
        special_tokens=[
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]"
        ]
    )

    # Train tokenizer if not already trained, and save it
    if not os.path.exists("tokenizer.json"):
        tokenizer.train(files=["data/sentences.txt"], trainer=trainer)
        tokenizer.save("tokenizer.json")

    return tokenizer


def create_and_train(config=Config(), device=None, tokenizer=None):

    # Tokenize the dataset and save the token ids
    if not os.path.exists("data/token_ids.json"):
        token_ids = tokenize_and_save(tokenizer, "data/sentences.txt", "data/token_ids.json")
    else:
        with open("data/token_ids.json", "r") as f:
            token_ids = json.load(f)

    # Load token ids and create datasets and dataloaders
    dataset = TextDataset(token_ids, context_len=64)

    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    # Set up the model and training
    vocab_size = tokenizer.get_vocab_size()
    print(f"Using device: {device}")

    config = Config()
    model = TinyStoriesLM(config)
    model.to(device)
    print(f"tokenizer vocab size: {vocab_size} Config vocab size: {config.vocab_size}")

    model.train()
    train(model, train_dataloader, val_dataloader, epochs=1, verbose=True)

    print("Training complete!")

    # save the model
    torch.save(model.state_dict(), "word_predictor_model.pth")

def predict_next_word(model, tokenizer, prompt, prefix, context_len=128, top_k = 100, device=None):
    model.eval()
    with torch.no_grad():
        encoded = tokenizer.encode(prompt)
        token_ids = encoded.ids[-context_len:]  # take last context_len tokens
        x = torch.tensor(token_ids).unsqueeze(0).to(device)  # shape (1, context_len)
        
        logits = model(x)  # shape (1, context_len, vocab_size)
        last_logits = logits[0, -1]  # shape (vocab_size,)
        top_k_indices = torch.topk(last_logits, top_k).indices.cpu().numpy()
        predicted_words = [tokenizer.decode([idx]) for idx in top_k_indices]
        predicted_words = [w for w in predicted_words if w.startswith(prefix)]

        return predicted_words
    

def test_model(config, device, tokenizer):
    model = TinyStoriesLM(config)
    model.load_state_dict(torch.load("word_predictor_model.pth", map_location=device))
    model.to(device)
    
    while True:
        prompt = input("> ")
        prefix = ""
        if not prompt.endswith(" "):
            prefix = prompt.strip().split()[-1]
            prompt = prompt[:-len(prefix)]


        print(f"Prompt: [{prompt}], Prefix: [{prefix}]")

        predictions = predict_next_word(model, tokenizer, prompt, prefix, context_len=config.block_size, device=device)
        print( predictions)



def main():
    tokenizer = get_tokenizer()
    config = Config()
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    create_and_train(config, device, tokenizer)

    test_model(config, device, tokenizer)
    



if __name__ == "__main__":
    main()