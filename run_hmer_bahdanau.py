"""
=======================================================================
HMER PIPELINE: CNN-BiLSTM + Bahdanau Attention + GRU Decoder
Dataset : HME100K (Handwritten Mathematical Expressions)
=======================================================================

HOW TO EXPLAIN THIS MODEL TO PROFESSORS (3 steps):

  STEP 1 — CNN Encoder
    "We pass the handwritten image through 4 convolutional blocks.
     Each block does: Conv -> BatchNorm -> ReLU -> MaxPool -> Dropout.
     We use asymmetric pooling in the last two blocks (reduce height
     more than width) so vertical structure — superscripts, fractions,
     square roots — is preserved in the feature map."

  STEP 2 — BiLSTM Context Layer
    "We flatten the 2D feature map into a sequence of spatial vectors
     and pass it through a Bidirectional LSTM. This lets the model
     capture context from both left-to-right and right-to-left,
     covering long-range dependencies across the image."

  STEP 3 — Bahdanau Attention + GRU Decoder
    "At each decoding step the GRU decoder has a hidden state s_t.
     We compute an attention score for every encoder position h_i:
         score(s_t, h_i) = v^T * tanh(W1*s_t + W2*h_i)
     Softmax over scores gives attention weights — the model learns
     WHERE TO LOOK in the image for the next token.
     The weighted sum of encoder features is the context c_t,
     combined with s_t to predict the next LaTeX token."

Architecture diagram:
  Image (H x W x 1)
      |
      v
  CNN Encoder (4 blocks with asymmetric pooling)
      |
      v  (B, H'*W', 256) flattened spatial sequence
  Bidirectional LSTM
      |
      v  (B, T_enc, 256) encoder features
  Bahdanau Attention  <--- decoder GRU state
      |
      v  context vector
  GRU Decoder  -->  LaTeX token predictions
=======================================================================
"""

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

print("=" * 65)
print("HMER: CNN-BiLSTM + BAHDANAU ATTENTION + GRU DECODER")
print("Dataset: HME100K  |  Simple & Explainable Architecture")
print("=" * 65)

# Environment & Hardware Setup
print("\nSystem Info:")
print("  Python     :", sys.executable)
print("  TensorFlow :", tf.__version__)
print("  PyTorch    :", torch.__version__)
print("  CUDA       :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  GPU        :", torch.cuda.get_device_name(0))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)

CHECKPOINT_DIR = os.path.abspath("./checkpoints_hmer")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print("Checkpoints Dir   :", CHECKPOINT_DIR)

# -----------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------
TRAIN_SAMPLES = 20000
VAL_SAMPLES   = 5000
TEST_SAMPLES  = 10000

IMG_HEIGHT = 64
IMG_WIDTH  = 512
MAX_TARGET_LENGTH = 150

# CNN Encoder
CNN_FILTERS  = [32, 64, 128, 256]   # filters per block
KERNEL_SIZE  = 3
CNN_DROPOUT  = 0.20

# BiLSTM (encoder context)
BILSTM_UNITS = 256   # total (128 forward + 128 backward)

# GRU Decoder
EMBED_DIM         = 256
DECODER_GRU_UNITS = 256
DECODER_DROPOUT   = 0.20

# Attention
ATTENTION_UNITS = 256

# Training
BATCH_SIZE              = 32
LEARNING_RATE           = 2e-4
EPOCHS                  = 8
EARLY_STOPPING_PATIENCE = 3
LR_REDUCTION_PATIENCE   = 2
LR_REDUCTION_FACTOR     = 0.5
MIN_LEARNING_RATE       = 1e-6
CLIP_NORM               = 5.0

# -----------------------------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------------------------
print("\n[1/6] Loading HME100K Dataset...")
dataset    = load_dataset("ConstantHao/HME100K")
full_train = dataset["train"]
full_test  = dataset["test"]

split      = full_train.train_test_split(test_size=0.10, seed=SEED)
train_pool = split["train"]
val_pool   = split["test"]

train_ds = train_pool.shuffle(seed=SEED).select(range(min(TRAIN_SAMPLES, len(train_pool))))
val_ds   = val_pool.shuffle(seed=SEED + 1).select(range(min(VAL_SAMPLES,   len(val_pool))))
test_ds  = full_test.shuffle(seed=SEED + 2).select(range(min(TEST_SAMPLES,  len(full_test))))

print(f"  Train : {len(train_ds):,} samples")
print(f"  Val   : {len(val_ds):,} samples")
print(f"  Test  : {len(test_ds):,} samples")

# -----------------------------------------------------------------------
# 2. Vocabulary
# -----------------------------------------------------------------------
print("\n[2/6] Building LaTeX Vocabulary...")
all_tokens = []
for split_ds in [train_ds, val_ds, test_ds]:
    for label in split_ds["label"]:
        all_tokens.extend(label.strip().split())

unique_tokens = sorted(set(all_tokens))

PAD_ID = 0   # padding — ignored in loss
UNK_ID = 1   # unknown token
BOS_ID = 2   # beginning of sequence
EOS_ID = 3   # end of sequence

token_to_id = {"<PAD>": PAD_ID, "<UNK>": UNK_ID, "<BOS>": BOS_ID, "<EOS>": EOS_ID}
for i, tok in enumerate(unique_tokens):
    token_to_id[tok] = i + 4

id_to_token = {v: k for k, v in token_to_id.items()}
NUM_CLASSES = len(token_to_id)
print(f"  Vocabulary size (incl. special tokens): {NUM_CLASSES}")

# -----------------------------------------------------------------------
# 3. Preprocessing
# -----------------------------------------------------------------------
def preprocess_image(image, augment=False):
    """
    Convert PIL image to normalized grayscale tensor of shape (H, W, 1).
    - Invert if background is light (white-on-black convention)
    - Resize to IMG_HEIGHT x IMG_WIDTH
    - Optional mild augmentation during training
    """
    image   = image.convert("L")
    img_arr = np.array(image, dtype=np.float32)
    if img_arr.mean() > 127:       # white background -> invert
        img_arr = 255.0 - img_arr
    img_arr    = img_arr / 255.0   # normalize [0, 1]
    img_tensor = tf.convert_to_tensor(img_arr[..., np.newaxis], dtype=tf.float32)
    img_tensor = tf.image.resize(img_tensor, [IMG_HEIGHT, IMG_WIDTH])
    if augment:
        img_tensor = tf.image.random_brightness(img_tensor, max_delta=0.08)
        img_tensor = tf.image.random_contrast(img_tensor, lower=0.85, upper=1.15)
        img_tensor = tf.clip_by_value(img_tensor, 0.0, 1.0)
    return img_tensor.numpy()

def encode_label(label_str, max_len=MAX_TARGET_LENGTH):
    """
    Convert a LaTeX string into integer token sequences.
    Returns:
      dec_input  : [<BOS>, tok1, tok2, ..., <PAD>]  (fed into decoder)
      dec_target : [tok1, tok2, ..., <EOS>, <PAD>]  (ground truth)
    """
    tokens = label_str.strip().split()
    ids    = [token_to_id.get(tok, UNK_ID) for tok in tokens]
    ids    = ids[:max_len - 2]
    dec_input  = [BOS_ID] + ids
    dec_target = ids + [EOS_ID]
    dec_input  += [PAD_ID] * (max_len - len(dec_input))
    dec_target += [PAD_ID] * (max_len - len(dec_target))
    return np.array(dec_input, dtype=np.int32), np.array(dec_target, dtype=np.int32)

def decode_ids(ids_list):
    """Convert a list of token IDs back to a LaTeX string."""
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
    """Infinite generator yielding (images, {decoder_input, decoder_target}) batches."""
    indices = np.arange(len(dataset_split))
    while True:
        np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx  = indices[start:start + batch_size]
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
                    "decoder_input":  np.stack(dec_inputs,  axis=0).astype(np.int32),
                    "decoder_target": np.stack(dec_targets, axis=0).astype(np.int32),
                }
            )

train_gen = data_generator(train_ds, BATCH_SIZE, augment=True)
val_gen   = data_generator(val_ds,   BATCH_SIZE, augment=False)
test_gen  = data_generator(test_ds,  BATCH_SIZE, augment=False)

# -----------------------------------------------------------------------
# 4. Model Architecture
# -----------------------------------------------------------------------
print("\n[3/6] Building CNN-BiLSTM + Bahdanau Attention + GRU Model...")


# --- 4a. Bahdanau Attention Layer ---
class BahdanauAttention(tf.keras.layers.Layer):
    """
    Classic Bahdanau (additive) attention.
    Reference: Bahdanau et al. 2015 - 'Neural Machine Translation by
               Jointly Learning to Align and Translate'

    Given:
      query  : decoder hidden state  s_t    shape (B, 1, decoder_units)
      values : encoder feature seq   h_i    shape (B, T_enc, encoder_units)

    Computes:
      score_i  = v^T * tanh(W1 * s_t + W2 * h_i)   <- one score per encoder pos
      alpha_i  = softmax(score_i)                    <- WHERE TO LOOK (weights)
      context  = sum_i(alpha_i * h_i)               <- weighted encoder features

    Easy to explain: 'The decoder decides how much attention to pay
    to each region of the image when predicting each token.'
    """
    def __init__(self, units=ATTENTION_UNITS, **kwargs):
        super().__init__(**kwargs)
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)  # applied to decoder state
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)  # applied to encoder features
        self.V  = tf.keras.layers.Dense(1,     use_bias=False)  # scoring vector

    def call(self, query, values):
        # query:   (B, seq_q, decoder_units)    e.g. (B, 150, 256) during training
        # values:  (B, T_enc, encoder_units)    e.g. (B, 512, 256)
        #
        # KEY FIX: we cannot add query and values directly when seq_q != T_enc.
        # Solution: expand dims so they broadcast against each other:
        #   W1(query) -> (B, seq_q, units)  expand -> (B, seq_q,    1,     units)
        #   W2(values)-> (B, T_enc, units)  expand -> (B,    1,  T_enc,    units)
        #   addition  ->                              (B, seq_q, T_enc, units)   ✓
        q = tf.expand_dims(self.W1(query),  axis=2)  # (B, seq_q, 1,     units)
        k = tf.expand_dims(self.W2(values), axis=1)  # (B, 1,     T_enc, units)
        score   = self.V(tf.nn.tanh(q + k))          # (B, seq_q, T_enc, 1)
        weights = tf.nn.softmax(score, axis=2)        # (B, seq_q, T_enc, 1)  softmax over encoder

        # Weighted sum over encoder positions
        # values: (B, T_enc, enc_units) -> expand -> (B, 1, T_enc, enc_units)
        context = tf.reduce_sum(
            weights * tf.expand_dims(values, axis=1),  # (B, seq_q, T_enc, enc_units)
            axis=2                                     # (B, seq_q, enc_units)
        )
        return context, tf.squeeze(weights, axis=-1)  # context: (B, seq_q, enc_units)


# --- 4b. CNN-BiLSTM Encoder ---
def build_cnn_bilstm_encoder():
    """
    Visual encoder:
      - 4 CNN blocks extract hierarchical visual features
      - Asymmetric pooling in blocks 3 & 4 preserves image HEIGHT
        (critical for 2D math: superscripts sit ABOVE base symbols,
         fractions have numerator OVER denominator, etc.)
      - BiLSTM adds left-right spatial context over the feature sequence

    Output shape: (B, H' * W', BILSTM_UNITS)
    With IMG_HEIGHT=64, IMG_WIDTH=512:
      After block1 (pool 2x2): 32 x 256
      After block2 (pool 2x2): 16 x 128
      After block3 (pool 2x1): 8  x 128  <- asymmetric: keep width
      After block4 (pool 2x1): 4  x 128  <- asymmetric: keep width
      Flatten: 4*128 = 512 spatial positions
    """
    inp = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image_input")

    # Block 1: Conv32 + Pool(2,2)
    x = tf.keras.layers.Conv2D(CNN_FILTERS[0], KERNEL_SIZE, padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(CNN_DROPOUT)(x)

    # Block 2: Conv64 + Pool(2,2)
    x = tf.keras.layers.Conv2D(CNN_FILTERS[1], KERNEL_SIZE, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(CNN_DROPOUT)(x)

    # Block 3: Conv128 + Pool(2,1)  <- ASYMMETRIC: reduce height, keep width
    # WHY? Math has vertical structure: x^2 has '2' ABOVE 'x'.
    # Keeping more width preserves these vertical positions.
    x = tf.keras.layers.Conv2D(CNN_FILTERS[2], KERNEL_SIZE, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1))(x)   # <- asymmetric
    x = tf.keras.layers.Dropout(CNN_DROPOUT)(x)

    # Block 4: Conv256 + Pool(2,1)  <- ASYMMETRIC again
    x = tf.keras.layers.Conv2D(CNN_FILTERS[3], KERNEL_SIZE, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1))(x)   # <- asymmetric
    x = tf.keras.layers.Dropout(CNN_DROPOUT)(x)

    # Flatten spatial dimensions H' and W' into one sequence dimension
    # (B, H', W', 256) -> (B, H'*W', 256)
    x = tf.keras.layers.Reshape((-1, CNN_FILTERS[3]))(x)

    # Project to BiLSTM input size
    x = tf.keras.layers.Dense(BILSTM_UNITS, use_bias=False)(x)

    # BiLSTM: reads spatial sequence forward AND backward
    # Output: (B, H'*W', BILSTM_UNITS)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(BILSTM_UNITS // 2, return_sequences=True)
    )(x)

    return tf.keras.Model(inputs=inp, outputs=x, name="CNN_BiLSTM_Encoder")

encoder = build_cnn_bilstm_encoder()
print("\n  --- Encoder Summary ---")
encoder.summary(line_length=70)


# --- 4c. GRU Decoder with Bahdanau Attention ---
class GRUDecoder(tf.keras.Model):
    """
    GRU-based decoder with Bahdanau Attention.

    At each decoding time step t:
      1. Embed previous token:   e_t = Embedding(y_{t-1})
      2. Compute attention:      c_t = Attention(s_{t-1}, encoder_features)
      3. Concatenate + GRU:      s_t = GRU([e_t ; c_t], s_{t-1})
      4. Predict next token:     y_t = Dense(s_t + c_t)

    WHY GRU instead of LSTM?
      GRU has 2 gates (reset + update) vs LSTM's 3 (+ cell state).
      Similar performance, simpler to explain and slightly faster.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(
            input_dim=NUM_CLASSES,
            output_dim=EMBED_DIM,
            mask_zero=False,
            name="token_embedding"
        )
        self.attention = BahdanauAttention(units=ATTENTION_UNITS, name="bahdanau_attention")
        self.gru = tf.keras.layers.GRU(
            DECODER_GRU_UNITS,
            return_sequences=True,
            return_state=True,
            name="decoder_gru"
        )
        self.dropout = tf.keras.layers.Dropout(DECODER_DROPOUT)
        self.fc = tf.keras.layers.Dense(NUM_CLASSES, name="output_projection")

    def call(self, decoder_input, encoder_features, training=False):
        """
        Teacher-forcing mode (used during training):
        All decoder input tokens fed at once.

        Args:
          decoder_input:    (B, seq_len) token IDs starting with <BOS>
          encoder_features: (B, T_enc, BILSTM_UNITS) from CNN-BiLSTM encoder
        Returns:
          logits: (B, seq_len, NUM_CLASSES)
        """
        emb        = self.embedding(decoder_input)              # (B, seq_len, EMBED_DIM)
        # context now shape (B, seq_len, BILSTM_UNITS) — one context per decoder step
        context, _ = self.attention(emb, encoder_features)      # (B, seq_len, BILSTM_UNITS)

        gru_in  = tf.concat([emb, context], axis=-1)   # (B, seq_len, EMBED_DIM + BILSTM_UNITS)
        gru_out, _ = self.gru(gru_in, training=training)        # (B, seq_len, GRU_UNITS)
        gru_out = self.dropout(gru_out, training=training)

        combined = gru_out + context                             # residual context connection
        logits   = self.fc(combined)                            # (B, seq_len, NUM_CLASSES)
        return logits


decoder = GRUDecoder(name="GRU_Decoder")

# Warm-up forward pass to build all weights
dummy_img    = tf.zeros([2, IMG_HEIGHT, IMG_WIDTH, 1], dtype=tf.float32)
dummy_tok    = tf.zeros([2, MAX_TARGET_LENGTH], dtype=tf.int32)
dummy_enc    = encoder(dummy_img, training=False)
dummy_logits = decoder(dummy_tok, dummy_enc, training=False)

enc_params   = int(np.sum([np.prod(v.shape) for v in encoder.trainable_variables]))
dec_params   = int(np.sum([np.prod(v.shape) for v in decoder.trainable_variables]))
total_params = enc_params + dec_params

print(f"\n  Encoder parameters : {enc_params:,}")
print(f"  Decoder parameters : {dec_params:,}")
print(f"  Total parameters   : {total_params:,}")
print("  [OK] Model built successfully")

# -----------------------------------------------------------------------
# 5. Loss Function
# -----------------------------------------------------------------------
def masked_sequence_loss(target, logits):
    """
    Cross-entropy loss that ignores PAD positions.
    We don't penalize predictions at padding steps.
    """
    loss = tf.keras.losses.sparse_categorical_crossentropy(target, logits, from_logits=True)
    mask = tf.cast(tf.not_equal(target, PAD_ID), dtype=loss.dtype)
    loss = loss * mask
    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)

# -----------------------------------------------------------------------
# 6. Optimizer & Checkpoints
# -----------------------------------------------------------------------
optimizer        = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM)
BEST_ENC_WEIGHTS = os.path.join(CHECKPOINT_DIR, "best_encoder.weights.h5")
BEST_DEC_WEIGHTS = os.path.join(CHECKPOINT_DIR, "best_decoder.weights.h5")

# -----------------------------------------------------------------------
# 7. Training Loop
# -----------------------------------------------------------------------
print("\n[4/6] Starting Training...")
steps_per_epoch  = len(train_ds) // BATCH_SIZE
validation_steps = len(val_ds)   // BATCH_SIZE

best_val_loss    = float("inf")
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    train_loss  = tf.keras.metrics.Mean()

    print(f"\n--- Epoch {epoch}/{EPOCHS} ---")

    for step in range(steps_per_epoch):
        images, targets = next(train_gen)
        dec_inp = targets["decoder_input"]
        dec_tgt = targets["decoder_target"]

        with tf.GradientTape() as tape:
            enc_features = encoder(images, training=True)
            logits       = decoder(dec_inp, enc_features, training=True)
            loss         = masked_sequence_loss(dec_tgt, logits)

        # Backprop through both encoder and decoder
        all_vars = encoder.trainable_variables + decoder.trainable_variables
        grads    = tape.gradient(loss, all_vars)
        optimizer.apply_gradients(zip(grads, all_vars))
        train_loss.update_state(loss)

        if (step + 1) % 50 == 0 or (step + 1) == steps_per_epoch:
            elapsed = (time.time() - epoch_start) / 60.0
            print(
                f"\r  Step {step+1:04d}/{steps_per_epoch:04d} "
                f"| Train Loss: {train_loss.result().numpy():.4f} "
                f"| Time: {elapsed:.1f}m",
                end=""
            )

    # Validation
    val_loss = tf.keras.metrics.Mean()
    for _ in range(validation_steps):
        images, targets = next(val_gen)
        dec_inp = targets["decoder_input"]
        dec_tgt = targets["decoder_target"]
        enc_features = encoder(images, training=False)
        logits       = decoder(dec_inp, enc_features, training=False)
        val_loss.update_state(masked_sequence_loss(dec_tgt, logits))

    v_res  = float(val_loss.result().numpy())
    t_res  = float(train_loss.result().numpy())
    cur_lr = float(optimizer.learning_rate.numpy())

    print(f"\n  Results -> Train Loss: {t_res:.4f} | Val Loss: {v_res:.4f} | LR: {cur_lr:.2e}")

    # Save best checkpoint
    if v_res < best_val_loss:
        best_val_loss    = v_res
        patience_counter = 0
        encoder.save_weights(BEST_ENC_WEIGHTS)
        decoder.save_weights(BEST_DEC_WEIGHTS)
        print(f"  [OK] Saved best checkpoint (val_loss={v_res:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= LR_REDUCTION_PATIENCE:
            new_lr = max(cur_lr * LR_REDUCTION_FACTOR, MIN_LEARNING_RATE)
            optimizer.learning_rate.assign(new_lr)
            print(f"  -> LR reduced to {new_lr:.2e}")

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"  Early stopping at epoch {epoch}")
        break

# -----------------------------------------------------------------------
# 8. Load Best Weights
# -----------------------------------------------------------------------
print("\n[5/6] Loading Best Weights for Evaluation...")
if os.path.exists(BEST_ENC_WEIGHTS) and os.path.exists(BEST_DEC_WEIGHTS):
    encoder.load_weights(BEST_ENC_WEIGHTS)
    decoder.load_weights(BEST_DEC_WEIGHTS)
    print("  [OK] Best checkpoint loaded.")
else:
    print("  [WARN] Checkpoint not found - using final epoch weights.")

# -----------------------------------------------------------------------
# 9. Greedy Decoding (for evaluation)
# -----------------------------------------------------------------------
def decode_greedy(raw_images, max_len=100):
    """
    Auto-regressive greedy decoding at inference time.
    Steps:
      1. Encode image with CNN-BiLSTM
      2. Start with <BOS>
      3. Each step: attention -> GRU -> pick highest-prob token
      4. Stop at <EOS> or max_len
    """
    processed  = np.stack([preprocess_image(img, augment=False) for img in raw_images], axis=0)
    batch_size = processed.shape[0]

    enc_features = encoder(processed, training=False)  # (B, T_enc, BILSTM_UNITS)

    current_tokens = tf.fill([batch_size, 1], tf.cast(BOS_ID, tf.int32))
    gru_state      = tf.zeros([batch_size, DECODER_GRU_UNITS], dtype=tf.float32)
    generated      = [[] for _ in range(batch_size)]
    active         = np.ones(batch_size, dtype=bool)

    for _ in range(max_len):
        emb            = decoder.embedding(current_tokens)          # (B, 1, EMBED_DIM)
        context, _     = decoder.attention(emb, enc_features)       # (B, 1, BILSTM_UNITS)
        gru_in         = tf.concat([emb, context], axis=-1)
        gru_out, gru_state = decoder.gru(gru_in, initial_state=gru_state, training=False)
        combined       = gru_out + context
        logits         = decoder.fc(combined)                        # (B, 1, NUM_CLASSES)
        next_tok       = tf.argmax(logits[:, 0, :], axis=-1, output_type=tf.int32).numpy()

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


# -----------------------------------------------------------------------
# 10. Evaluation Helpers
# -----------------------------------------------------------------------
def canonicalize_latex(s):
    """Normalize LaTeX for fair comparison (ignore style-only differences)."""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace("{ ", "{").replace(" }", "}")
    s = s.replace(" ^ ", "^").replace(" _ ", "_")
    return s.strip()

def compute_levenshtein(seq1, seq2):
    """Token-level edit distance."""
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

print("\n[6/6] Evaluating on Test Set...")
EVAL_SAMPLES    = min(1000, len(test_ds))
EVAL_BATCH_SIZE = 32

test_preds, test_truths = [], []
for start in range(0, EVAL_SAMPLES, EVAL_BATCH_SIZE):
    end         = min(start + EVAL_BATCH_SIZE, EVAL_SAMPLES)
    batch_items = [test_ds[i] for i in range(start, end)]
    preds       = decode_greedy([x["image"] for x in batch_items])
    test_preds.extend(preds)
    test_truths.extend([x["label"] for x in batch_items])
    if (start // EVAL_BATCH_SIZE + 1) % 5 == 0:
        print(f"  Evaluated {end}/{EVAL_SAMPLES} samples...")

# Compute metrics
raw_exact, canon_exact = 0, 0
token_accuracies, similarities, edit_distances = [], [], []

for pred, true in zip(test_preds, test_truths):
    p_toks = pred.strip().split()
    t_toks = true.strip().split()

    if pred.strip() == true.strip():
        raw_exact += 1
    if canonicalize_latex(pred) == canonicalize_latex(true):
        canon_exact += 1

    matches = sum(p == t for p, t in zip(p_toks, t_toks))
    tok_acc = matches / max(len(t_toks), len(p_toks), 1)
    token_accuracies.append(tok_acc)

    ed  = compute_levenshtein(p_toks, t_toks)
    sim = max(0.0, 1.0 - ed / max(len(t_toks), 1))
    edit_distances.append(ed)
    similarities.append(sim)

# Results
print("\n" + "=" * 65)
print("FINAL EVALUATION  —  CNN-BiLSTM + Bahdanau Attn + GRU")
print("=" * 65)
print(f"  Test Samples Evaluated     : {EVAL_SAMPLES:,}")
print(f"  Raw Exact Match Accuracy   : {100.0 * raw_exact   / EVAL_SAMPLES:.2f}%")
print(f"  Canonical Exact Match Acc  : {100.0 * canon_exact / EVAL_SAMPLES:.2f}%")
print(f"  Token-Level Accuracy       : {100.0 * np.mean(token_accuracies):.2f}%")
print(f"  Normalized Token Similarity: {100.0 * np.mean(similarities):.2f}%")
print(f"  Average Edit Distance      : {np.mean(edit_distances):.2f} tokens")
print("=" * 65)

# Save results to JSON
results = {
    "model": "CNN-BiLSTM + Bahdanau Attention + GRU Decoder",
    "dataset": "HME100K",
    "eval_samples": EVAL_SAMPLES,
    "raw_exact_match_pct":       round(100.0 * raw_exact   / EVAL_SAMPLES, 2),
    "canonical_exact_match_pct": round(100.0 * canon_exact / EVAL_SAMPLES, 2),
    "token_accuracy_pct":        round(100.0 * float(np.mean(token_accuracies)), 2),
    "token_similarity_pct":      round(100.0 * float(np.mean(similarities)), 2),
    "avg_edit_distance":         round(float(np.mean(edit_distances)), 3),
}
results_path = os.path.join(CHECKPOINT_DIR, "results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to : {results_path}")
print("Checkpoints saved to:", CHECKPOINT_DIR)
print("\nDONE!")
