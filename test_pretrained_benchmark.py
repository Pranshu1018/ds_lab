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
from datasets import load_dataset
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 70)
print("PRETRAINED TrOCR vs. CUSTOM 2D-COVERAGE CNN-HMER BENCHMARK")
print("=" * 70)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Compute Device     :", DEVICE)
if torch.cuda.is_available():
    print("GPU Hardware       :", torch.cuda.get_device_name(0))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 1. Load Dataset
print("\n[1/4] Loading HME100K Dataset from local cache...")
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

# For benchmarking inference speed, evaluate on 500 samples
EVAL_SAMPLES = 500
test_eval = test_sub.select(range(EVAL_SAMPLES))

# 2. Reconstruct Vocabulary identical to training
print("\n[2/4] Constructing Vocabulary...")
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

# 3. Model Architecture
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

# 4. Load Custom Model Checkpoint
print("\n[3/4] Loading Custom 2D-Coverage Model Checkpoint...")
custom_model = PyTorchHMERModel().to(DEVICE)
ckpt_path = os.path.abspath("./checkpoints_hmer/pytorch_hmer_best.pt")
if os.path.exists(ckpt_path):
    custom_model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    print("  [OK] Successfully loaded weights from:", ckpt_path)
else:
    print("  [ERROR] Checkpoint not found at:", ckpt_path)
    sys.exit(1)
custom_model.eval()

# 5. Load Pretrained TrOCR Model
print("\n[4/4] Loading Pretrained TrOCR Model (microsoft/trocr-small-stage1)...")
trocr_model_name = "microsoft/trocr-small-stage1"
processor = TrOCRProcessor.from_pretrained(trocr_model_name)
trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_model_name).to(DEVICE)
trocr_model.eval()
print(f"  [OK] Pretrained TrOCR Loaded on {DEVICE}")

# 6. Evaluation on Test Expressions
print(f"\nRunning Benchmark Evaluation on {len(test_eval)} Unseen Test Expressions...")

custom_preds = []
trocr_preds = []
ground_truths = []

batch_size = 16
num_batches = math.ceil(len(test_eval) / batch_size)

t0 = time.time()
for b in range(num_batches):
    batch_items = [test_eval[i] for i in range(b * batch_size, min((b + 1) * batch_size, len(test_eval)))]
    images_pil = [item["image"] for item in batch_items]
    labels = [item["label"] for item in batch_items]
    ground_truths.extend(labels)

    # A. Custom Model Inference
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

    # B. Pretrained TrOCR Inference
    pixel_values = processor(images=[img.convert("RGB") for img in images_pil], return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        generated_ids = trocr_model.generate(pixel_values, max_new_tokens=64)
    trocr_decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    trocr_preds.extend(trocr_decoded)

    sys.stdout.write(f"\r  Progress: {len(custom_preds)}/{len(test_sub)} completed...")
    sys.stdout.flush()

total_time = time.time() - t0
print(f"\n  Inference Completed in {total_time:.2f}s!")

# 7. Metrics & Display
custom_res = evaluate_metrics(custom_preds, ground_truths)
trocr_res = evaluate_metrics(trocr_preds, ground_truths)

print("\n" + "=" * 75)
print("FINAL BENCHMARK COMPARISON TABLE (HME100K UNSEEN TEST EXPRESSIONS)")
print("=" * 75)
print(f"{'Evaluation Metric':<32} | {'Custom 2D-Coverage CNN':<22} | {'Pretrained TrOCR':<18}")
print("-" * 75)
print(f"{'Raw Exact Expression Acc (%)':<32} | {custom_res['raw_exact']:<22.2f} | {trocr_res['raw_exact']:<18.2f}")
print(f"{'Canonical Exact Match Acc (%)':<32} | {custom_res['canon_exact']:<22.2f} | {trocr_res['canon_exact']:<18.2f}")
print(f"{'Token-Level Accuracy (%)':<32} | {custom_res['token_acc']:<22.2f} | {trocr_res['token_acc']:<18.2f}")
print(f"{'Normalized Token Similarity (%)':<32} | {custom_res['similarity']:<22.2f} | {trocr_res['similarity']:<18.2f}")
print(f"{'Average Edit Distance (Lower=Best)':<32} | {custom_res['avg_edit_dist']:<22.2f} | {trocr_res['avg_edit_dist']:<18.2f}")
print("=" * 75)

print("\n--- Side-by-Side Sample Inspection ---")
for i in range(min(5, len(ground_truths))):
    print(f"\n[Test Expression #{i+1}]")
    print(f"  Ground Truth    : {ground_truths[i]}")
    print(f"  Custom Model    : {custom_preds[i]}")
    print(f"  Pretrained TrOCR: {trocr_preds[i]}")

print("\n" + "=" * 75)
print("BENCHMARK EVALUATION FINISHED SUCCESSFULLY!")
print("=" * 75)
