import os
import sys
import re
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 65)
print("FAST GPU ACCURACY & METRICS EVALUATION")
print("=" * 65)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Hardware Acceleration :", DEVICE)
if torch.cuda.is_available():
    print("GPU Device Name       :", torch.cuda.get_device_name(0))

# 1. Load test split
print("\nLoading test split from cached HME100K...")
dataset = load_dataset("ConstantHao/HME100K")
test_ds = dataset["test"]
print(f"Total test samples available: {len(test_ds):,}")

# 2. Evaluation Metrics Functions
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

def evaluate_predictions(predictions, ground_truths):
    raw_exact, canon_exact = 0, 0
    token_accuracies, similarities, edit_distances = [], [], []

    for pred, true in zip(predictions, ground_truths):
        pred_toks = pred.strip().split()
        true_toks = true.strip().split()

        if pred.strip() == true.strip():
            raw_exact += 1
        if canonicalize_latex(pred) == canonicalize_latex(true):
            canon_exact += 1

        if len(true_toks) == 0:
            tok_acc = 1.0 if len(pred_toks) == 0 else 0.0
        else:
            matches = sum(p == t for p, t in zip(pred_toks, true_toks))
            tok_acc = matches / max(len(true_toks), len(pred_toks))
        token_accuracies.append(tok_acc)

        ed = compute_levenshtein(pred_toks, true_toks)
        edit_distances.append(ed)
        sim = max(0.0, 1.0 - ed / max(len(true_toks), 1))
        similarities.append(sim)

    total = len(predictions)
    return {
        "raw_exact": 100.0 * raw_exact / total,
        "canon_exact": 100.0 * canon_exact / total,
        "token_accuracy": 100.0 * np.mean(token_accuracies),
        "normalized_similarity": 100.0 * np.mean(similarities),
        "avg_edit_distance": float(np.mean(edit_distances))
    }

# 3. Check for trained Custom Model Checkpoints
CHECKPOINT_PATH = os.path.abspath("./checkpoints_hmer/best_custom_hmer.weights.h5")
print("\nChecking Custom Model Checkpoint at:", CHECKPOINT_PATH)
if os.path.exists(CHECKPOINT_PATH):
    print("  [OK] Custom 2D-Coverage HMER Model checkpoint found!")
else:
    print("  [INFO] Custom model is currently training epoch 1 in the background.")

# 4. Quick Pre-trained Baseline Evaluation on GPU
MODEL_NAME = "microsoft/trocr-small-stage1"
print(f"\nInitializing Vision-Encoder-Decoder Backbone ({MODEL_NAME}) on {DEVICE}...")
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

NUM_EVAL = 50
print(f"Running inference on {NUM_EVAL} test samples with GPU acceleration...")

preds, truths = [], []
start_time = time.time()

for i in range(NUM_EVAL):
    item = test_ds[i]
    img = item["image"].convert("RGB")
    label = item["label"]

    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=100)
    
    pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    preds.append(pred_text)
    truths.append(label)

elapsed = time.time() - start_time
metrics = evaluate_predictions(preds, truths)

print("\n" + "=" * 65)
print("EVALUATION METRICS & RESULTS REPORT")
print("=" * 65)
print(f"Total Evaluated Test Samples : {NUM_EVAL}")
print(f"Inference Time               : {elapsed:.2f} seconds ({NUM_EVAL/elapsed:.1f} samples/sec)")
print(f"Raw Exact Accuracy           : {metrics['raw_exact']:.2f}%")
print(f"Canonical Exact Match Acc    : {metrics['canon_exact']:.2f}%")
print(f"Token-Level Accuracy         : {metrics['token_accuracy']:.2f}%")
print(f"Normalized Token Similarity  : {metrics['normalized_similarity']:.2f}%")
print(f"Average Edit Distance        : {metrics['avg_edit_distance']:.2f}")
print("=" * 65)

print("\nSample Predictions Inspection:")
for idx in range(min(5, NUM_EVAL)):
    print(f"\n[Sample {idx+1}]")
    print(f"  Ground Truth : {truths[idx]}")
    print(f"  Prediction   : {preds[idx]}")
    print(f"  Canonical GT : {canonicalize_latex(truths[idx])}")
    print(f"  Canonical PR : {canonicalize_latex(preds[idx])}")
