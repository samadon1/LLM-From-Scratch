import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import random

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride, augment_prob=0.1):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        self.augment_prob = augment_prob

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids, target_ids = self.input_ids[idx], self.target_ids[idx]
        if random.random() < self.augment_prob:
            input_ids = self.augment_sequence(input_ids)
        return input_ids, target_ids

    def augment_sequence(self, sequence):
        mask_idx = random.randint(0, len(sequence) - 1)
        sequence[mask_idx] = self.tokenizer.encode("[MASK]")[0]
        return sequence

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.relative_position_encoding = nn.Parameter(torch.randn(2 * context_length - 1, self.head_dim))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        relative_position = self._get_relative_positions(num_tokens)
        rel_attn_scores = self._relative_attention_scores(queries, relative_position)
        attn_scores += rel_attn_scores

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec

    def _get_relative_positions(self, length):
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(0).repeat(length, 1)
        distance_mat = range_mat - range_mat.T
        return distance_mat + length - 1

    def _relative_attention_scores(self, queries, relative_position):
        embeddings = self.relative_position_encoding[relative_position]
        return torch.einsum('bhld,lrd->bhlr', queries, embeddings)

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            Swish(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        self.conv = nn.Conv1d(cfg["emb_dim"], cfg["emb_dim"], kernel_size=3, padding=1)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)

        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.token_type_emb = nn.Embedding(cfg["num_token_types"], cfg["emb_dim"])

        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, token_type_ids=None):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(in_idx)
        token_type_embeds = self.token_type_emb(token_type_ids)
        
        x = tok_embeds + pos_embeds + token_type_embeds
        x = self.drop_emb(x)
        
        for block in self.trf_blocks:
            x = block(x)
        
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def focal_loss(logits, targets, alpha=0.25, gamma=2):
    ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1-pt)**gamma * ce_loss
    return focal_loss.mean()

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = focal_loss(logits.flatten(0, 1), target_batch.flatten())
    return loss

def top_k_sampling(logits, k=10):
    v, _ = torch.topk(logits, k)
    logits[logits < v[:, [-1]]] = float('-inf')
    probas = torch.softmax(logits, dim=-1)
    return probas

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = top_k_sampling(logits)
        idx_next = torch.multinomial(probas, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def visualize_attention(attention_weights):
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='viridis')
    plt.colorbar()
    plt.title("Attention Weights")
    plt.show()
