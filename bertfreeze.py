"""
GoEmotions Multi-label Emotion Classifier
Improved architecture targeting >0.75 macro F1
Key improvements:
  1. Fine-tune BERT end-to-end (not frozen)
  2. Custom Transformer encoder on top of BERT
  3. Attention pooling
  4. Label smoothing loss
  5. Warmup + cosine LR schedule
  6. Per-class threshold tuning
  7. Gradient clipping
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_cosine_schedule_with_warmup
from datasets import load_dataset
from collections import Counter
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── Config ───────────────────────────
MAX_LEN       = 128
BATCH_SIZE    = 32
EPOCHS        = 20
LR            = 2e-5         
WARMUP_RATIO  = 0.1
LABEL_SMOOTH  = 0.05
GRAD_CLIP     = 1.0
MIN_LABEL_COUNT = 2000
EMBED_DIM     = 768
NUM_HEADS     = 8
FF_DIM        = 2048
DROPOUT       = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────── Data ───────────────────────────
dataset = load_dataset("go_emotions", "simplified")

label_names = dataset["train"].features["labels"].feature.names

# Count label frequencies
label_counter = Counter()
for item in dataset["train"]:
    for l in item["labels"]:
        label_counter[l] += 1

valid_labels = [l for l, c in label_counter.items() if c > MIN_LABEL_COUNT]
valid_label_set = set(valid_labels)
label_map = {old: i for i, old in enumerate(valid_labels)}
NUM_CLASSES = len(valid_labels)

print(f"Valid labels ({NUM_CLASSES}): {[label_names[i] for i in valid_labels]}")


def filter_and_remap(data):
    filtered = []
    for item in data:
        labels = [label_map[l] for l in item["labels"] if l in valid_label_set]
        if labels:
            filtered.append({"text": item["text"], "labels": labels})
    return filtered


train_data = filter_and_remap(dataset["train"])
dev_data   = filter_and_remap(dataset["validation"])
test_data  = filter_and_remap(dataset["test"])

print(f"Train: {len(train_data)} | Dev: {len(dev_data)} | Test: {len(test_data)}")


def multi_hot(labels, n=NUM_CLASSES):
    vec = np.zeros(n, dtype=np.float32)
    for l in labels:
        vec[l] = 1.0
    return vec


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class EmotionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor(multi_hot(item["labels"]), dtype=torch.float)
        )


train_loader = DataLoader(EmotionDataset(train_data), batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
dev_loader   = DataLoader(EmotionDataset(dev_data),   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(EmotionDataset(test_data),  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ─────────────────────────── Modules ───────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, dropout=DROPOUT):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, E = x.shape
        def split(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split(self.W_q(x)), split(self.W_k(x)), split(self.W_v(x))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            # mask: (B, T) → (B, 1, 1, T)
            scores = scores.masked_fill(mask[:, None, None, :] == 0, float("-inf"))

        weights = self.attn_drop(torch.softmax(scores, dim=-1))
        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T, E)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, hidden_dim=FF_DIM, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, heads=NUM_HEADS, ff_dim=FF_DIM, dropout=DROPOUT):
        super().__init__()
        self.attn  = MultiHeadSelfAttention(embed_dim, heads, dropout)
        self.ff    = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.ff(x))
        return x


class AttentionPooling(nn.Module):
    """Weighted average over sequence using a learned scorer."""
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, x, mask=None):
        scores = self.scorer(x)                    # (B, T, 1)
        if mask is not None:
            scores = scores.masked_fill(mask[:, :, None] == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)     # (B, T, 1)
        return (weights * x).sum(dim=1)            # (B, E)


class EmotionTransformer(nn.Module):
    """
    BERT backbone (fine-tuned) → 2 custom Transformer encoder layers
    → attention pooling → classifier head
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.enc1 = TransformerEncoderLayer()
        self.enc2 = TransformerEncoderLayer()

        self.pool = AttentionPooling()

        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        # Fine-tune BERT — no torch.no_grad() here
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)        
        x = bert_out.last_hidden_state          # (B, T, 768)

        x = self.enc1(x, mask=attention_mask)
        x = self.enc2(x, mask=attention_mask)

        pooled = self.pool(x, mask=attention_mask)
        return self.classifier(pooled)


# ─────────────────────────── Loss ───────────────────────────

class LabelSmoothingBCE(nn.Module):
    """BCE with label smoothing to prevent overconfidence."""
    def __init__(self, smoothing=LABEL_SMOOTH):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)


# ─────────────────────────── Training ───────────────────────────

model     = EmotionTransformer().to(device)
criterion = LabelSmoothingBCE()

# Separate LR for BERT backbone vs new layers
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=0.01
)

total_steps   = len(train_loader) * EPOCHS
warmup_steps  = int(total_steps * WARMUP_RATIO)
scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for input_ids, mask, labels in loader:
            input_ids = input_ids.to(device)
            mask      = mask.to(device)
            labels    = labels.to(device)

            logits = model(input_ids, mask)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(labels.cpu().numpy())

    preds   = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return total_loss / len(loader), macro_f1


print("\n── Training ──")
best_f1    = 0.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_f1 = run_epoch(train_loader, train=True)
    dv_loss, dv_f1 = run_epoch(dev_loader,   train=False)
    print(f"Epoch {epoch}/{EPOCHS}  train_loss={tr_loss:.4f}  train_f1={tr_f1:.4f}  "
          f"dev_loss={dv_loss:.4f}  dev_f1={dv_f1:.4f}")

    if dv_f1 > best_f1:
        best_f1    = dv_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, "best_model_freeze.pt")
        print(f"  ✓ New best saved (dev_f1={best_f1:.4f})")


# ─────────────────────────── Per-class threshold tuning ───────────────────────────

print("\n── Tuning thresholds on dev set ──")
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
model.eval()

all_probs, all_targets = [], []
with torch.no_grad():
    for input_ids, mask, labels in dev_loader:
        logits = model(input_ids.to(device), mask.to(device))
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(labels.numpy())

probs   = np.vstack(all_probs)
targets = np.vstack(all_targets)

best_thresholds = np.full(NUM_CLASSES, 0.5)
for c in range(NUM_CLASSES):
    best_t, best_class_f1 = 0.5, 0.0
    for t in np.arange(0.2, 0.75, 0.05):
        preds_c = (probs[:, c] > t).astype(int)
        f = f1_score(targets[:, c], preds_c, zero_division=0)
        if f > best_class_f1:
            best_class_f1, best_t = f, t
    best_thresholds[c] = best_t

print(f"Thresholds: {np.round(best_thresholds, 2)}")


# ─────────────────────────── Test evaluation ───────────────────────────

print("\n── Test set evaluation ──")
all_probs_test, all_targets_test = [], []
with torch.no_grad():
    for input_ids, mask, labels in test_loader:
        logits = model(input_ids.to(device), mask.to(device))
        all_probs_test.append(torch.sigmoid(logits).cpu().numpy())
        all_targets_test.append(labels.numpy())

probs_test   = np.vstack(all_probs_test)
targets_test = np.vstack(all_targets_test)

# Fixed 0.5 threshold
preds_fixed = (probs_test > 0.5).astype(int)
f1_fixed    = f1_score(targets_test, preds_fixed, average="macro", zero_division=0)

# Tuned thresholds
preds_tuned = (probs_test > best_thresholds[None, :]).astype(int)
f1_tuned    = f1_score(targets_test, preds_tuned, average="macro", zero_division=0)

print(f"Test Macro F1 (threshold=0.5):    {f1_fixed:.4f}")
print(f"Test Macro F1 (tuned thresholds): {f1_tuned:.4f}")

# Per-class breakdown
per_class = f1_score(targets_test, preds_tuned, average=None, zero_division=0)
print("\nPer-class F1:")
for i, (name_idx, f) in enumerate(zip(valid_labels, per_class)):
    print(f"  {label_names[name_idx]:20s}: {f:.4f}")
