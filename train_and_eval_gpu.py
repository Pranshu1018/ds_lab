import os
import sys
import re
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 65)
print("GPU-ACCELERATED HMER MODEL TRAINING & ACCURACY EVALUATION")
print("=" * 65)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Compute Device     :", DEVICE)
if torch.cuda.is_available():
    print("GPU Hardware       :", torch.cuda.get_device_name(0))
    print(f"GPU VRAM Available : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

CHECKPOINT_DIR = os.path.abspath("./checkpoints_hmer")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 1. Load Dataset
print("\n[1/5] Loading HME100K Dataset from local cache...")
dataset = load_dataset("ConstantHao/HME100K")
full_train = dataset["train"]
full_test  = dataset["test"]

TRAIN_SAMPLES = 20000
VAL_SAMPLES   = 2000
TEST_SAMPLES  = 2000

split = full_train.train_test_split(test_size=0.10, seed=SEED)
train_pool = split["train"]
val_pool   = split["test"]

train_sub = train_pool.shuffle(seed=SEED).select(range(min(TRAIN_SAMPLES, len(train_pool))))
val_sub   = val_pool.shuffle(seed=SEED + 1).select(range(min(VAL_SAMPLES, len(val_pool))))
test_sub  = full_test.shuffle(seed=SEED + 2).select(range(min(TEST_SAMPLES, len(full_test))))

print(f"  Train Subsplit : {len(train_sub):,}")
print(f"  Val Subsplit   : {len(val_sub):,}")
print(f"  Test Subsplit  : {len(test_sub):,}")

# 2. Vocabulary
print("\n[2/5] Constructing Vocabulary...")
all_tokens = []
for split_ds in [train_sub, val_sub, test_sub]:
    for label in split_ds["label"]:
        all_tokens.extend(label.strip().split())

unique_tokens = sorted(set(all_tokens))
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
token_to_id = {"<PAD>": PAD_ID, "<UNK>": UNK_ID, "<BOS>": BOS_ID, "<EOS>": EOS_ID}
for i, tok in enumerate(unique_tokens):
    token_to_id[tok] = i + 4
id_to_token = {v: k for k, v in token_to_id.items()}
VOCAB_SIZE = len(token_to_id)
print(f"  Vocabulary Size : {VOCAB_SIZE} unique classes")

# 3. PyTorch Dataset & Transform
IMG_HEIGHT, IMG_WIDTH = 64, 512
MAX_LEN = 120

def process_img(pil_img):
    img = pil_img.convert("L").resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img, dtype=np.float32)
    if arr.mean() > 127:
        arr = 255.0 - arr
    arr = arr / 255.0
    return torch.tensor(arr).unsqueeze(0) # (1, H, W)

class HMERDataset(Dataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img_t = process_img(item["image"])
        toks = item["label"].strip().split()
        ids = [token_to_id.get(t, UNK_ID) for t in toks][:MAX_LEN - 2]
        
        dec_in = [BOS_ID] + ids
        dec_out = ids + [EOS_ID]
        
        dec_in += [PAD_ID] * (MAX_LEN - len(dec_in))
        dec_out += [PAD_ID] * (MAX_LEN - len(dec_out))
        
        return (
            img_t,
            torch.tensor(dec_in, dtype=torch.long),
            torch.tensor(dec_out, dtype=torch.long),
            item["label"]
        )

train_loader = DataLoader(HMERDataset(train_sub), batch_size=32, shuffle=True, pin_memory=True, num_workers=0)
val_loader   = DataLoader(HMERDataset(val_sub), batch_size=32, shuffle=False, pin_memory=True, num_workers=0)
test_loader  = DataLoader(HMERDataset(test_sub), batch_size=32, shuffle=False, pin_memory=True, num_workers=0)

# 4. PyTorch 2D Coverage CNN-HMER Architecture
class ScaledDotProductAttention2D(nn.Module):
    def __init__(self, d_model=256, coverage_kernel=5):
        super().__init__()
        self.W_q = nn.Linear(384, d_model, bias=False)
        self.W_k = nn.Linear(256, d_model, bias=False)
        self.W_v = nn.Linear(256, d_model, bias=False)
        self.W_cov = nn.Conv1d(1, 1, kernel_size=coverage_kernel, padding=coverage_kernel//2, bias=False)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, query, values, coverage=None):
        # query: (B, Dec_Len, 384)
        # values: (B, Spatial_Points, 256)
        # coverage: (B, Spatial_Points, 1)
        Q = self.W_q(query)   # (B, Dec_Len, d_model)
        K = self.W_k(values)  # (B, Spatial_Points, d_model)
        V = self.W_v(values)  # (B, Spatial_Points, d_model)

        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale # (B, Dec_Len, Spatial_Points)

        if coverage is not None:
            # (B, Spatial_Points, 1) -> (B, 1, Spatial_Points)
            cov_bias = self.W_cov(coverage.transpose(1, 2)) # (B, 1, Spatial_Points)
            scores = scores + cov_bias

        attn_weights = F.softmax(scores, dim=-1) # (B, Dec_Len, Spatial_Points)
        context = torch.bmm(attn_weights, V)     # (B, Dec_Len, d_model)
        return context, attn_weights

class PyTorchHMERModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=256, dec_units=384):
        super().__init__()
        # Asymmetric CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Block 1: 64x512 -> 32x256
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.1),

            # Block 2: 32x256 -> 16x128
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.1),

            # Block 3: 16x128 -> 8x128 (Asymmetric)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(0.1),

            # Block 4: 8x128 -> 4x128
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(0.1)
        )

        self.proj = nn.Linear(256, d_model, bias=False)
        self.spatial_lstm = nn.LSTM(d_model, d_model // 2, batch_first=True, bidirectional=True)
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.decoder_lstm = nn.LSTM(d_model, dec_units, batch_first=True)
        self.attention = ScaledDotProductAttention2D(d_model=d_model)
        
        self.classifier = nn.Sequential(
            nn.Linear(dec_units + d_model, dec_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(dec_units, vocab_size)
        )

    def encode(self, images):
        # images: (B, 1, 64, 512)
        features = self.cnn(images) # (B, 256, 4, 128)
        B, C, H, W = features.shape
        x_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, C) # (B, 512, 256)
        x_proj = self.proj(x_flat)
        enc_out, _ = self.spatial_lstm(x_proj) # (B, 512, 256)
        return enc_out

    def forward(self, images, dec_in):
        enc_features = self.encode(images)
        emb = self.embedding(dec_in)
        dec_out, _ = self.decoder_lstm(emb)
        context, _ = self.attention(dec_out, enc_features)
        combined = torch.cat([dec_out, context], dim=-1)
        logits = self.classifier(combined)
        return logits

print("\n[3/5] Instantiating PyTorch Model on GPU...")
model = PyTorchHMERModel().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Model Trainable Parameters: {total_params:,}")

# 5. Training Loop on GPU with FP16
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

EPOCHS = 25
print(f"\n[4/5] Training Custom 2D-Coverage Model for {EPOCHS} Epochs on {DEVICE}...")

best_val_loss = float("inf")
best_model_path = os.path.join(CHECKPOINT_DIR, "pytorch_hmer_best.pt")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0
    start_t = time.time()

    for step, (imgs, dec_in, dec_out, _) in enumerate(train_loader):
        imgs = imgs.to(DEVICE, non_blocking=True)
        dec_in = dec_in.to(DEVICE, non_blocking=True)
        dec_out = dec_out.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            logits = model(imgs, dec_in)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), dec_out.reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        total_train_loss += loss.item()

    # Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for imgs, dec_in, dec_out, _ in val_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            dec_in = dec_in.to(DEVICE, non_blocking=True)
            dec_out = dec_out.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                logits = model(imgs, dec_in)
                v_loss = criterion(logits.reshape(-1, VOCAB_SIZE), dec_out.reshape(-1))
            total_val_loss += v_loss.item()

    avg_train = total_train_loss / len(train_loader)
    avg_val   = total_val_loss / len(val_loader)
    epoch_time = time.time() - start_t

    print(f"  Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Time: {epoch_time:.1f}s")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), best_model_path)
        print(f"    [OK] Saved Best Checkpoint -> Val Loss: {avg_val:.4f}")

# 6. Evaluation on Test Set
print("\n[5/5] Running Complete Evaluation on Test Split...")
model.load_state_dict(torch.load(best_model_path))
model.eval()

def canonicalize_latex(latex_str):
    s = latex_str.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace("{ ", "{").replace(" }", "}")
    s = s.replace(" ^ ", "^").replace(" _ ", "_")
    return s.strip()

def compute_levenshtein(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp
    return dp[m]

def decode_greedy_gpu(model, images, max_len=100):
    B = images.shape[0]
    enc_features = model.encode(images)
    spatial_points = enc_features.shape[1]

    curr_tokens = torch.full((B, 1), BOS_ID, dtype=torch.long, device=DEVICE)
    h = torch.zeros(1, B, 384, device=DEVICE)
    c = torch.zeros(1, B, 384, device=DEVICE)
    coverage = torch.zeros(B, spatial_points, 1, device=DEVICE)

    generated = [[] for _ in range(B)]
    active = np.ones(B, dtype=bool)

    for _ in range(max_len):
        emb = model.embedding(curr_tokens) # (B, 1, 256)
        dec_out, (h, c) = model.decoder_lstm(emb, (h, c))
        context, attn_w = model.attention(dec_out, enc_features, coverage=coverage)
        coverage = coverage + attn_w.transpose(1, 2)

        combined = torch.cat([dec_out, context], dim=-1)
        logits = model.classifier(combined)
        next_tok = torch.argmax(logits[:, 0, :], dim=-1) # (B,)
        next_np = next_tok.cpu().numpy()

        for b in range(B):
            if not active[b]:
                continue
            tid = int(next_np[b])
            if tid in (EOS_ID, PAD_ID):
                active[b] = False
            else:
                generated[b].append(tid)

        if not np.any(active):
            break
        curr_tokens = next_tok.unsqueeze(1)

    return [
        " ".join([id_to_token.get(t, "<UNK>") for t in seq])
        for seq in generated
    ]

all_preds, all_truths = [], []
eval_start = time.time()

with torch.no_grad():
    for imgs, _, _, raw_labels in test_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        preds = decode_greedy_gpu(model, imgs, max_len=MAX_LEN)
        all_preds.extend(preds)
        all_truths.extend(raw_labels)

raw_exact, canon_exact = 0, 0
tok_accs, sims, edit_dists = [], [], []

for pred, true in zip(all_preds, all_truths):
    p_toks = pred.strip().split()
    t_toks = true.strip().split()

    if pred.strip() == true.strip():
        raw_exact += 1
    if canonicalize_latex(pred) == canonicalize_latex(true):
        canon_exact += 1

    matches = sum(p == t for p, t in zip(p_toks, t_toks))
    tok_accs.append(matches / max(len(t_toks), len(p_toks), 1))
    
    ed = compute_levenshtein(p_toks, t_toks)
    edit_dists.append(ed)
    sims.append(max(0.0, 1.0 - ed / max(len(t_toks), 1)))

total_test = len(all_preds)
eval_time = time.time() - eval_start

print("\n" + "=" * 70)
print("CUSTOM 2D-COVERAGE CNN-HMER TEST EVALUATION RESULTS")
print("=" * 70)
print(f"Evaluated Test Samples       : {total_test:,}")
print(f"Inference Time               : {eval_time:.1f}s ({total_test/eval_time:.1f} images/sec on GPU)")
print(f"Raw Exact Expression Acc     : {100.0 * raw_exact / total_test:.2f}%")
print(f"Canonical Exact Match Acc    : {100.0 * canon_exact / total_test:.2f}%")
print(f"Token-Level Accuracy         : {100.0 * np.mean(tok_accs):.2f}%")
print(f"Normalized Token Similarity  : {100.0 * np.mean(sims):.2f}%")
print(f"Average Edit Distance        : {np.mean(edit_dists):.2f}")
print("=" * 70)

print("\nSample Predictions Inspection:")
for idx in range(min(5, total_test)):
    print(f"\n[Test Example {idx+1}]")
    print(f"  Ground Truth : {all_truths[idx]}")
    print(f"  Model Pred   : {all_preds[idx]}")
    print(f"  Canonical GT : {canonicalize_latex(all_truths[idx])}")
    print(f"  Canonical PR : {canonicalize_latex(all_preds[idx])}")
print("\n" + "=" * 70)
print("EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 70)
