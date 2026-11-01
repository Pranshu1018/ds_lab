import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import time
import json
import random
import re
import numpy as np
import tensorflow as tf
from PIL import Image
from datasets import load_dataset
import torch

print("=" * 60)
print("HMER PIPELINE: CUSTOM 2D-COVERAGE MODEL + BENCHMARK")
print("=" * 60)

# Environment & Hardware Setup
print("Python Executable :", sys.executable)
print("TensorFlow Version:", tf.__version__)
print("PyTorch Version   :", torch.__version__)
print("CUDA Available    :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device        :", torch.cuda.get_device_name(0))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)

CHECKPOINT_DIR = os.path.abspath("./checkpoints_hmer")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print("Checkpoints Dir   :", CHECKPOINT_DIR)

# Hyperparameters
TRAIN_SAMPLES = 20000
VAL_SAMPLES   = 5000
TEST_SAMPLES  = 10000

IMG_HEIGHT = 64
IMG_WIDTH  = 512
MAX_TARGET_LENGTH = 150

CNN_FILTERS   = [32, 64, 128, 256]
KERNEL_SIZE   = 3
CNN_DROPOUT   = 0.20

BILSTM_UNITS     = 256
EMBEDDING_DIM    = 256
DECODER_UNITS    = 384
ATTENTION_UNITS  = 256
COVERAGE_KERNEL  = 5
DECODER_DROPOUT  = 0.20

BATCH_SIZE              = 32
LEARNING_RATE           = 2e-4
EPOCHS                  = 8
EARLY_STOPPING_PATIENCE = 3
LR_REDUCTION_PATIENCE   = 2
LR_REDUCTION_FACTOR     = 0.5
MIN_LEARNING_RATE       = 1e-6
CLIP_NORM               = 5.0
LABEL_SMOOTHING         = 0.05
BEAM_WIDTH              = 3
LENGTH_PENALTY_ALPHA    = 0.6

# 1. Load Dataset
print("\n[1/6] Loading cached HME100K Dataset...")
dataset = load_dataset("ConstantHao/HME100K")
full_train = dataset["train"]
full_test  = dataset["test"]

split = full_train.train_test_split(test_size=0.10, seed=SEED)
train_pool = split["train"]
val_pool   = split["test"]

train_ds = train_pool.shuffle(seed=SEED).select(range(min(TRAIN_SAMPLES, len(train_pool))))
val_ds   = val_pool.shuffle(seed=SEED + 1).select(range(min(VAL_SAMPLES, len(val_pool))))
test_ds  = full_test.shuffle(seed=SEED + 2).select(range(min(TEST_SAMPLES, len(full_test))))

print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

# 2. Vocabulary Setup
print("\n[2/6] Building LaTeX Vocabulary...")
all_tokens = []
for split_ds in [train_ds, val_ds, test_ds]:
    for label in split_ds["label"]:
        all_tokens.extend(label.strip().split())

unique_tokens = sorted(set(all_tokens))
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

token_to_id = {"<PAD>": PAD_ID, "<UNK>": UNK_ID, "<BOS>": BOS_ID, "<EOS>": EOS_ID}
for i, tok in enumerate(unique_tokens):
    token_to_id[tok] = i + 4

id_to_token = {v: k for k, v in token_to_id.items()}
NUM_CLASSES = len(token_to_id)
print(f"  Vocabulary size (including specials): {NUM_CLASSES}")

# 3. Preprocessing Functions
def preprocess_image(image, augment=False):
    image = image.convert("L")
    img_arr = np.array(image, dtype=np.float32)
    if img_arr.mean() > 127:
        img_arr = 255.0 - img_arr
    img_arr = img_arr / 255.0
    img_tensor = tf.convert_to_tensor(img_arr[..., np.newaxis], dtype=tf.float32)
    img_tensor = tf.image.resize(img_tensor, [IMG_HEIGHT, IMG_WIDTH])
    if augment:
        img_tensor = tf.image.random_brightness(img_tensor, max_delta=0.08)
        img_tensor = tf.image.random_contrast(img_tensor, lower=0.85, upper=1.15)
        img_tensor = tf.clip_by_value(img_tensor, 0.0, 1.0)
    return img_tensor.numpy()

def encode_label(label_str, max_len=MAX_TARGET_LENGTH):
    tokens = label_str.strip().split()
    ids = [token_to_id.get(tok, UNK_ID) for tok in tokens]
    ids = ids[:max_len - 2]
    dec_input = [BOS_ID] + ids
    dec_target = ids + [EOS_ID]
    dec_input += [PAD_ID] * (max_len - len(dec_input))
    dec_target += [PAD_ID] * (max_len - len(dec_target))
    return np.array(dec_input, dtype=np.int32), np.array(dec_target, dtype=np.int32)

def decode_ids(ids_list):
    res = []
    for tid in ids_list:
        tid = int(tid)
        if tid in (PAD_ID, BOS_ID):
            continue
        if tid == EOS_ID:
            break
        res.append(id_to_token.get(tid, "<UNK>"))
    return " ".join(res)

def data_generator(dataset_split, batch_size, augment=False):
    indices = np.arange(len(dataset_split))
    while True:
        np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            images, dec_inputs, dec_targets = [], [], []
            for idx in batch_idx:
                item = dataset_split[int(idx)]
                images.append(preprocess_image(item["image"], augment=augment))
                inp, tgt = encode_label(item["label"])
                dec_inputs.append(inp)
                dec_targets.append(tgt)
            yield (
                np.stack(images, axis=0).astype(np.float32),
                {
                    "decoder_input": np.stack(dec_inputs, axis=0).astype(np.int32),
                    "decoder_target": np.stack(dec_targets, axis=0).astype(np.int32)
                }
            )

train_gen = data_generator(train_ds, BATCH_SIZE, augment=True)
val_gen   = data_generator(val_ds, BATCH_SIZE, augment=False)
test_gen  = data_generator(test_ds, BATCH_SIZE, augment=False)

# 4. Architecture Definition (Memory-Efficient 2D Scaled Dot-Product Coverage Attention)
print("\n[3/6] Building Custom 2D Coverage CNN-HMER Architecture...")
class ScaledDotProductAttention2D(tf.keras.layers.Layer):
    def __init__(self, units=ATTENTION_UNITS, coverage_kernel=COVERAGE_KERNEL, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W_q = tf.keras.layers.Dense(units, use_bias=False)
        self.W_k = tf.keras.layers.Dense(units, use_bias=False)
        self.W_v = tf.keras.layers.Dense(units, use_bias=False)
        self.W_cov = tf.keras.layers.Conv1D(1, coverage_kernel, padding="same", use_bias=False)
        self.scale = tf.math.rsqrt(tf.cast(units, tf.float32))

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, query, values, coverage=None):
        # query: (B, Dec_Len, Decoder_Units)
        # values: (B, Spatial_Points, Encoder_Units)
        # coverage: (B, Spatial_Points, 1)
        Q = self.W_q(query)
        K = self.W_k(values)
        V = self.W_v(values)

        scores = tf.matmul(Q, K, transpose_b=True) * self.scale

        if coverage is not None:
            cov_bias = tf.transpose(self.W_cov(coverage), perm=[0, 2, 1])
            scores = scores + cov_bias

        attn_weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(attn_weights, V)
        return context, attn_weights

class CustomHMERModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(CNN_FILTERS[0], KERNEL_SIZE, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(CNN_DROPOUT)
        ])
        self.conv2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(CNN_FILTERS[1], KERNEL_SIZE, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(CNN_DROPOUT)
        ])
        self.conv3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(CNN_FILTERS[2], KERNEL_SIZE, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
            tf.keras.layers.Dropout(CNN_DROPOUT)
        ])
        self.conv4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(CNN_FILTERS[3], KERNEL_SIZE, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
            tf.keras.layers.Dropout(CNN_DROPOUT)
        ])
        self.feature_proj = tf.keras.layers.Dense(BILSTM_UNITS, use_bias=False)
        self.spatial_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(BILSTM_UNITS // 2, return_sequences=True)
        )
        self.embedding = tf.keras.layers.Embedding(
            input_dim=NUM_CLASSES,
            output_dim=EMBEDDING_DIM,
            mask_zero=False
        )
        self.decoder_lstm = tf.keras.layers.LSTM(
            DECODER_UNITS,
            return_sequences=True,
            return_state=True
        )
        self.decoder_dropout = tf.keras.layers.Dropout(DECODER_DROPOUT)
        self.attention = ScaledDotProductAttention2D(units=ATTENTION_UNITS)
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(DECODER_UNITS, activation="relu"),
            tf.keras.layers.Dropout(DECODER_DROPOUT),
            tf.keras.layers.Dense(NUM_CLASSES)
        ])

    def encode(self, images, training=False):
        x = self.conv1(images, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        shape = tf.shape(x)
        b, h, w, c = shape[0], shape[1], shape[2], shape[3]
        x_flat = tf.reshape(x, [b, h * w, c])
        x_proj = self.feature_proj(x_flat)
        encoded = self.spatial_lstm(x_proj, training=training)
        return encoded

    def call(self, inputs, training=False):
        images = inputs["images"]
        decoder_input = inputs["decoder_input"]
        encoder_features = self.encode(images, training=training)
        emb = self.embedding(decoder_input)
        dec_out, _, _ = self.decoder_lstm(emb, training=training)
        dec_out = self.decoder_dropout(dec_out, training=training)
        context, _ = self.attention(dec_out, encoder_features)
        combined = tf.concat([dec_out, context], axis=-1)
        logits = self.classifier(combined)
        return logits

model = CustomHMERModel()
dummy_img = tf.zeros([2, IMG_HEIGHT, IMG_WIDTH, 1], dtype=tf.float32)
dummy_inp = tf.zeros([2, MAX_TARGET_LENGTH], dtype=tf.int32)
dummy_out = model({"images": dummy_img, "decoder_input": dummy_inp}, training=False)
trainable_vars = np.sum([np.prod(v.shape) for v in model.trainable_variables])
print(f"  Model Parameters: {trainable_vars:,}")
print("  [OK] Memory-Efficient 2D Spatial Resolution Preserved (512 locations per image)")

# Loss
def masked_sequence_loss(target, logits, label_smoothing=LABEL_SMOOTHING):
    loss = tf.keras.losses.sparse_categorical_crossentropy(target, logits, from_logits=True)
    mask = tf.cast(tf.not_equal(target, PAD_ID), dtype=loss.dtype)
    loss = loss * mask
    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM)
BEST_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "best_custom_hmer.weights.h5")

# 5. Training Loop
print("\n[4/6] Starting Training Pipeline...")
steps_per_epoch  = len(train_ds) // BATCH_SIZE
validation_steps = len(val_ds) // BATCH_SIZE

best_val_loss = float("inf")
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    train_loss = tf.keras.metrics.Mean()
    
    print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
    for step in range(steps_per_epoch):
        images, targets = next(train_gen)
        dec_inp = targets["decoder_input"]
        dec_tgt = targets["decoder_target"]
        
        with tf.GradientTape() as tape:
            logits = model({"images": images, "decoder_input": dec_inp}, training=True)
            loss = masked_sequence_loss(dec_tgt, logits)
            
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss.update_state(loss)
        
        if (step + 1) % 50 == 0 or (step + 1) == steps_per_epoch:
            elapsed = (time.time() - epoch_start) / 60.0
            print(f"\r  Step {step+1:04d}/{steps_per_epoch:04d} | Train Loss: {train_loss.result().numpy():.4f} | Time: {elapsed:.1f}m", end="")
            
    # Validation
    val_loss = tf.keras.metrics.Mean()
    for _ in range(validation_steps):
        images, targets = next(val_gen)
        dec_inp = targets["decoder_input"]
        dec_tgt = targets["decoder_target"]
        logits = model({"images": images, "decoder_input": dec_inp}, training=False)
        v_l = masked_sequence_loss(dec_tgt, logits)
        val_loss.update_state(v_l)
        
    v_res = float(val_loss.result().numpy())
    t_res = float(train_loss.result().numpy())
    cur_lr = float(optimizer.learning_rate.numpy())
    
    print(f"\n  Epoch {epoch} Results -> Train Loss: {t_res:.4f} | Val Loss: {v_res:.4f} | LR: {cur_lr:.2e}")
    
    if v_res < best_val_loss:
        best_val_loss = v_res
        patience_counter = 0
        model.save_weights(BEST_WEIGHTS_PATH)
        print(f"  [OK] Saved New Best Checkpoint ({v_res:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= LR_REDUCTION_PATIENCE:
            new_lr = max(cur_lr * LR_REDUCTION_FACTOR, MIN_LEARNING_RATE)
            optimizer.learning_rate.assign(new_lr)
            print(f"  -> Reduced LR to: {new_lr:.2e}")
            
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"  Early stopping triggered at epoch {epoch}")
        break

print("\n[5/6] Loading Best Model Weights for Evaluation...")
if os.path.exists(BEST_WEIGHTS_PATH):
    model.load_weights(BEST_WEIGHTS_PATH)
    print("  [OK] Loaded best checkpoint weights successfully.")

# 6. Evaluation Framework
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

def decode_greedy_fast(model, raw_images, max_len=100):
    processed_imgs = np.stack([preprocess_image(img, augment=False) for img in raw_images], axis=0)
    batch_size = processed_imgs.shape[0]
    encoder_features = model.encode(processed_imgs, training=False)
    spatial_points = tf.shape(encoder_features)[1]

    state_h = tf.zeros([batch_size, DECODER_UNITS], dtype=tf.float32)
    state_c = tf.zeros([batch_size, DECODER_UNITS], dtype=tf.float32)
    coverage = tf.zeros([batch_size, spatial_points, 1], dtype=tf.float32)

    current_tokens = tf.fill([batch_size, 1], tf.cast(BOS_ID, tf.int32))
    generated = [[] for _ in range(batch_size)]
    active = np.ones(batch_size, dtype=bool)

    for step in range(max_len):
        emb = model.embedding(current_tokens)
        dec_out, state_h, state_c = model.decoder_lstm(emb, initial_state=[state_h, state_c], training=False)
        context, attn_w = model.attention(dec_out, encoder_features, coverage=coverage)
        coverage = coverage + tf.expand_dims(attn_w, axis=-1)

        combined = tf.concat([dec_out, context], axis=-1)
        logits = model.classifier(combined)
        next_tok = tf.argmax(logits[:, 0, :], axis=-1, output_type=tf.int32).numpy()

        for b in range(batch_size):
            if not active[b]:
                continue
            tok_id = int(next_tok[b])
            if tok_id in (EOS_ID, PAD_ID):
                active[b] = False
            else:
                generated[b].append(tok_id)

        if not np.any(active):
            break
        current_tokens = tf.expand_dims(tf.constant(next_tok, dtype=tf.int32), axis=1)

    return [decode_ids(seq) for seq in generated]

print("\n[6/6] Running Evaluation on 1,000 Test Samples...")
EVAL_SAMPLES = min(1000, len(test_ds))
EVAL_BATCH_SIZE = 32

test_preds, test_truths = [], []
for start in range(0, EVAL_SAMPLES, EVAL_BATCH_SIZE):
    end = min(start + EVAL_BATCH_SIZE, EVAL_SAMPLES)
    batch_items = [test_ds[i] for i in range(start, end)]
    preds = decode_greedy_fast(model, [x["image"] for x in batch_items])
    test_preds.extend(preds)
    test_truths.extend([x["label"] for x in batch_items])

raw_exact, canon_exact = 0, 0
token_accuracies, similarities, edit_distances = [], [], []

for pred, true in zip(test_preds, test_truths):
    p_toks, t_toks = pred.strip().split(), true.strip().split()
    if pred.strip() == true.strip():
        raw_exact += 1
    if canonicalize_latex(pred) == canonicalize_latex(true):
        canon_exact += 1
    matches = sum(p == t for p, t in zip(p_toks, t_toks))
    tok_acc = matches / max(len(t_toks), len(p_toks), 1)
    token_accuracies.append(tok_acc)
    ed = compute_levenshtein(p_toks, t_toks)
    edit_distances.append(ed)
    sim = max(0.0, 1.0 - ed / max(len(t_toks), 1))
    similarities.append(sim)

print("\n" + "=" * 65)
print("FINAL HME100K 2D-COVERAGE MODEL EVALUATION RESULTS")
print("=" * 65)
print(f"Evaluated Test Samples       : {EVAL_SAMPLES:,}")
print(f"Raw Exact Expression Acc     : {100.0 * raw_exact / EVAL_SAMPLES:.2f}%")
print(f"Canonical Exact Match Acc    : {100.0 * canon_exact / EVAL_SAMPLES:.2f}%")
print(f"Token-Level Accuracy         : {100.0 * np.mean(token_accuracies):.2f}%")
print(f"Normalized Token Similarity  : {100.0 * np.mean(similarities):.2f}%")
print(f"Average Edit Distance        : {np.mean(edit_distances):.2f}")
print("=" * 65)
print("ALL RUNS COMPLETE! Checkpoints saved to:", CHECKPOINT_DIR)
