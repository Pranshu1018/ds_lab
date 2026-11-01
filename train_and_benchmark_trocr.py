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
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 80)
print("FAST 3-WAY HMER BENCHMARK: ZERO-SHOT vs FINE-TUNED TrOCR vs CUSTOM CNN")
print("=" * 80)

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
print("\n[1/5] Loading HME100K Dataset...")
dataset = load_dataset("ConstantHao/HME100K")
full_train = dataset["train"]
full_test  = dataset["test"]

TRAIN_SAMPLES = 5000  # Fast, optimal fine-tuning sample size
VAL_SAMPLES   = 500
TEST_SAMPLES  = 300

split = full_train.train_test_split(test_size=0.10, seed=SEED)
train_pool = split["train"]
val_pool   = split["test"]

train_sub = train_pool.shuffle(seed=SEED).select(range(min(TRAIN_SAMPLES, len(train_pool))))
val_sub   = val_pool.shuffle(seed=SEED + 1).select(range(min(VAL_SAMPLES, len(val_pool))))
test_eval = full_test.shuffle(seed=SEED + 2).select(range(min(TEST_SAMPLES, len(full_test))))

print(f"  TrOCR Fine-Tune Train : {len(train_sub):,}")
print(f"  TrOCR Fine-Tune Val   : {len(val_sub):,}")
print(f"  Benchmark Test Size   : {len(test_eval):,}")

# 2. Reconstruct Vocabulary for Custom Model (matching trained checkpoint)
all_tokens_for_custom = []
vocab_train_sub = train_pool.shuffle(seed=SEED).select(range(min(20000, len(train_pool))))
vocab_val_sub   = val_pool.shuffle(seed=SEED + 1).select(range(min(2000, len(val_pool))))
vocab_test_sub  = full_test.shuffle(seed=SEED + 2).select(range(min(2000, len(full_test))))
for s_ds in [vocab_train_sub, vocab_val_sub, vocab_test_sub]:
    for l in s_ds["label"]:
        all_tokens_for_custom.extend(l.strip().split())

unique_tokens = sorted(set(all_tokens_for_custom))
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
token_to_id = {"<PAD>": PAD_ID, "<UNK>": UNK_ID, "<BOS>": BOS_ID, "<EOS>": EOS_ID}
for i, tok in enumerate(unique_tokens):
    token_to_id[tok] = i + 4
id_to_token = {v: k for k, v in token_to_id.items()}
VOCAB_SIZE = len(token_to_id)

IMG_HEIGHT, IMG_WIDTH = 64, 512

class ScaledDotProductAttention2D(nn.Module):
    def __init__(self, d_model=256, coverage_kernel=5):
        super().__init__()
        self.W_q = nn.Linear(384, d_model, bias=False)
        self.W_k = nn.Linear(256, d_model, bias=False)
        self.W_v = nn.Linear(256, d_model, bias=False)
        self.W_cov = nn.Conv1d(1, 1, kernel_size=coverage_kernel, padding=coverage_kernel//2, bias=False)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, query, values, coverage=None):
        Q = self.W_q(query)
        K = self.W_k(values)
        V = self.W_v(values)
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        if coverage is not None:
            cov_bias = self.W_cov(coverage.transpose(1, 2))
            scores = scores + cov_bias
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, V)
        return context, attn_weights

class PyTorchHMERModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=256, dec_units=384):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(0.1),

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
        features = self.cnn(images)
        B, C, H, W = features.shape
        x_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_proj = self.proj(x_flat)
        enc_out, _ = self.spatial_lstm(x_proj)
        return enc_out

def process_img(pil_img):
    img = pil_img.convert("L").resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img, dtype=np.float32)
    if arr.mean() > 127:
        arr = 255.0 - arr
    arr = arr / 255.0
    return torch.tensor(arr).unsqueeze(0)

# 3. TrOCR Fast Data Transforms
print("\n[2/5] Initializing Fast TrOCR Pipeline...")
model_name = "microsoft/trocr-small-stage1"
processor = TrOCRProcessor.from_pretrained(model_name)

trocr_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class FastTrOCRDataset(Dataset):
    def __init__(self, hf_ds, tokenizer, max_len=64):
        self.images = [item["image"].convert("RGB") for item in hf_ds]
        labels_raw  = [item["label"] for item in hf_ds]
        
        # Pre-tokenize all labels in 1 fast batch
        tok_out = tokenizer(
            labels_raw,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt"
        )
        self.labels = tok_out.input_ids
        self.labels[self.labels == tokenizer.pad_token_id] = -100

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_tensor = trocr_transform(self.images[idx])
        return img_tensor, self.labels[idx]

train_trocr_loader = DataLoader(FastTrOCRDataset(train_sub, processor.tokenizer), batch_size=16, shuffle=True, pin_memory=True)
val_trocr_loader   = DataLoader(FastTrOCRDataset(val_sub, processor.tokenizer), batch_size=16, shuffle=False, pin_memory=True)

trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name).to(DEVICE)
trocr_model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
trocr_model.config.pad_token_id = processor.tokenizer.pad_token_id
trocr_model.config.vocab_size = trocr_model.config.decoder.vocab_size

# Freeze vision encoder to accelerate fine-tuning by 3x on GPU
for p in trocr_model.encoder.parameters():
    p.requires_grad = False

# 4. Fine-Tuning TrOCR Decoder Head (2 Epochs)
print("\n[3/5] Fine-Tuning TrOCR Decoder on HME100K (Fast GPU AMP)...")
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, trocr_model.parameters()), lr=1e-4, weight_decay=1e-4)
scaler = torch.amp.GradScaler('cuda' if DEVICE.type == 'cuda' else 'cpu')

EPOCHS = 2
trocr_best_path = os.path.join(CHECKPOINT_DIR, "trocr_finetuned_best.pt")

for epoch in range(1, EPOCHS + 1):
    trocr_model.train()
    total_loss = 0.0
    t_start = time.time()
    
    for step, (p_vals, lbls) in enumerate(train_trocr_loader):
        p_vals = p_vals.to(DEVICE, non_blocking=True)
        lbls   = lbls.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda' if DEVICE.type == 'cuda' else 'cpu'):
            outputs = trocr_model(pixel_values=p_vals, labels=lbls)
            loss = outputs.loss
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trocr_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_trocr_loader)
    elapsed = time.time() - t_start
    print(f"  TrOCR Epoch {epoch}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

try:
    trocr_model.save_pretrained(os.path.join(CHECKPOINT_DIR, "trocr_finetuned"))
    print(f"  [OK] Saved Fine-Tuned TrOCR Model -> {os.path.join(CHECKPOINT_DIR, 'trocr_finetuned')}")
except Exception as e:
    print(f"  [NOTE] Skipping disk save: {e} (Model is loaded in GPU memory)")

# 5. Full 3-Way Comparative Evaluation
print(f"\n[4/5] Evaluating All 3 Models on {len(test_eval)} Unseen Test Expressions...")

# A. Custom Model
custom_model = PyTorchHMERModel().to(DEVICE)
ckpt_path = os.path.join(CHECKPOINT_DIR, "pytorch_hmer_best.pt")
custom_model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
custom_model.eval()

# B. Zero-Shot TrOCR
zero_shot_trocr = VisionEncoderDecoderModel.from_pretrained(model_name).to(DEVICE)
zero_shot_trocr.eval()

# C. Fine-Tuned TrOCR
trocr_model.eval()

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

def evaluate_metrics(preds, truths):
    raw_exact = 0
    canon_exact = 0
    token_correct = 0
    total_tokens = 0
    total_sim = 0.0
    total_dist = 0.0
    N = len(preds)

    for p, g in zip(preds, truths):
        p_clean = p.strip()
        g_clean = g.strip()
        if p_clean == g_clean:
            raw_exact += 1

        p_can = canonicalize_latex(p)
        g_can = canonicalize_latex(g)
        if p_can == g_can:
            canon_exact += 1

        p_toks = p.split()
        g_toks = g.split()
        min_len = min(len(p_toks), len(g_toks))
        token_correct += sum(1 for a, b in zip(p_toks[:min_len], g_toks[:min_len]) if a == b)
        total_tokens += max(len(g_toks), 1)

        dist = compute_levenshtein(p_toks, g_toks)
        max_l = max(len(p_toks), len(g_toks), 1)
        sim = max(0.0, 1.0 - dist / max_l)
        total_sim += sim
        total_dist += dist

    return {
        "raw_exact": (raw_exact / N) * 100,
        "canon_exact": (canon_exact / N) * 100,
        "token_acc": (token_correct / total_tokens) * 100,
        "similarity": (total_sim / N) * 100,
        "avg_edit_dist": total_dist / N
    }

custom_preds = []
zero_shot_preds = []
finetuned_trocr_preds = []
ground_truths = []

batch_size = 16
num_batches = math.ceil(len(test_eval) / batch_size)

t0 = time.time()
for b in range(num_batches):
    batch_items = [test_eval[i] for i in range(b * batch_size, min((b + 1) * batch_size, len(test_eval)))]
    images_pil = [item["image"] for item in batch_items]
    labels = [item["label"] for item in batch_items]
    ground_truths.extend(labels)

    # 1. Custom 2D-Coverage CNN
    custom_tensors = torch.stack([process_img(img) for img in images_pil]).to(DEVICE)
    with torch.no_grad():
        enc_f = custom_model.encode(custom_tensors)
        B_size = enc_f.shape[0]
        sp_pts = enc_f.shape[1]
        c_toks = torch.full((B_size, 1), BOS_ID, dtype=torch.long, device=DEVICE)
        h = torch.zeros(1, B_size, 384, device=DEVICE)
        c = torch.zeros(1, B_size, 384, device=DEVICE)
        cov = torch.zeros(B_size, sp_pts, 1, device=DEVICE)
        gen = [[] for _ in range(B_size)]
        active = np.ones(B_size, dtype=bool)

        for _ in range(100):
            emb = custom_model.embedding(c_toks)
            dec_out, (h, c) = custom_model.decoder_lstm(emb, (h, c))
            ctx, att_w = custom_model.attention(dec_out, enc_f, coverage=cov)
            cov = cov + att_w.transpose(1, 2)
            comb = torch.cat([dec_out, ctx], dim=-1)
            logits = custom_model.classifier(comb)
            next_t = torch.argmax(logits[:, -1, :], dim=-1)

            c_toks = next_t.unsqueeze(1)
            for idx_b in range(B_size):
                if active[idx_b]:
                    tid = next_t[idx_b].item()
                    if tid == EOS_ID:
                        active[idx_b] = False
                    elif tid not in (PAD_ID, BOS_ID):
                        gen[idx_b].append(id_to_token.get(tid, ""))
            if not active.any():
                break
        custom_preds.extend([" ".join(g) for g in gen])

    # TrOCR Images
    pixel_values = torch.stack([trocr_transform(img.convert("RGB")) for img in images_pil]).to(DEVICE)

    # 2. Zero-Shot TrOCR
    with torch.no_grad():
        gen_zero = zero_shot_trocr.generate(pixel_values, max_new_tokens=64)
    zero_shot_preds.extend(processor.batch_decode(gen_zero, skip_special_tokens=True))

    # 3. Fine-Tuned TrOCR
    with torch.no_grad():
        gen_ft = trocr_model.generate(pixel_values, max_new_tokens=64)
    finetuned_trocr_preds.extend(processor.batch_decode(gen_ft, skip_special_tokens=True))

    sys.stdout.write(f"\r  Evaluating: {len(custom_preds)}/{len(test_eval)} completed...")
    sys.stdout.flush()

print(f"\n  Inference Completed in {time.time() - t0:.1f}s!")

# 6. Comparative Benchmark Table
custom_res = evaluate_metrics(custom_preds, ground_truths)
zero_res   = evaluate_metrics(zero_shot_preds, ground_truths)
ft_res     = evaluate_metrics(finetuned_trocr_preds, ground_truths)

print("\n" + "=" * 90)
print("FINAL 3-WAY SCIENTIFIC BENCHMARK (HME100K UNSEEN EXPRESSIONS)")
print("=" * 90)
print(f"{'Evaluation Metric':<32} | {'Zero-Shot TrOCR':<16} | {'Fine-Tuned TrOCR':<18} | {'Custom 2D-CNN Model':<18}")
print("-" * 90)
print(f"{'Raw Exact Expression Acc (%)':<32} | {zero_res['raw_exact']:<16.2f} | {ft_res['raw_exact']:<18.2f} | {custom_res['raw_exact']:<18.2f}")
print(f"{'Canonical Exact Match Acc (%)':<32} | {zero_res['canon_exact']:<16.2f} | {ft_res['canon_exact']:<18.2f} | {custom_res['canon_exact']:<18.2f}")
print(f"{'Token-Level Accuracy (%)':<32} | {zero_res['token_acc']:<16.2f} | {ft_res['token_acc']:<18.2f} | {custom_res['token_acc']:<18.2f}")
print(f"{'Normalized Token Similarity (%)':<32} | {zero_res['similarity']:<16.2f} | {ft_res['similarity']:<18.2f} | {custom_res['similarity']:<18.2f}")
print(f"{'Average Edit Distance (Lower=Best)':<32} | {zero_res['avg_edit_dist']:<16.2f} | {ft_res['avg_edit_dist']:<18.2f} | {custom_res['avg_edit_dist']:<18.2f}")
print("=" * 90)

print("\n--- 3-Way Sample Prediction Inspection ---")
for i in range(min(5, len(ground_truths))):
    print(f"\n[Test Expression #{i+1}]")
    print(f"  Ground Truth     : {ground_truths[i]}")
    print(f"  Custom 2D-CNN    : {custom_preds[i]}")
    print(f"  Fine-Tuned TrOCR : {finetuned_trocr_preds[i]}")
    print(f"  Zero-Shot TrOCR  : {zero_shot_preds[i]}")

print("\n" + "=" * 90)
print("BENCHMARK COMPLETED SUCCESSFULLY!")
print("=" * 90)
