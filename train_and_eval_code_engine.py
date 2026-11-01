import os
import sys
import re
import ast
import time
import math
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 80)
print("ZERO-DISK-FOOTPRINT CODE & MATH EXPRESSION RECOGNITION & EXECUTION ENGINE")
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

# 1. Procedural Template Engine for Deterministic In-Memory Generation
CODE_TEMPLATES = [
    # 1. Quadratic & Polynomials
    ("y = {a} * x ** 2 + {b} * x + {c}", lambda a, b, c: {"inputs": [2, -1, 3], "func": lambda x: a * x**2 + b * x + c}),
    ("res = {a} * x ** 3 - {b} * x ** 2 + {c} * x - {d}", lambda a, b, c, d: {"inputs": [1, 2, -2], "func": lambda x: a*x**3 - b*x**2 + c*x - d}),
    ("f = lambda x: {a} * x ** 2 + {b} * math.sin(x)", lambda a, b: {"inputs": [0.5, 1.0, 2.0], "func": lambda x: a*x**2 + b*math.sin(x)}),
    
    # 2. Math library calls & square roots
    ("y = math.sqrt({a} * x + {b}) + {c} * math.cos(x)", lambda a, b, c: {"inputs": [1, 4, 9], "func": lambda x: math.sqrt(a*x + b) + c*math.cos(x)}),
    ("ans = math.exp({a} * x) / (1 + math.exp({a} * x))", lambda a: {"inputs": [0, 1, -1], "func": lambda x: math.exp(a*x) / (1 + math.exp(a*x))}),
    ("val = math.log({a} * x + {b}) + {c} * x", lambda a, b, c: {"inputs": [1, 2, 5], "func": lambda x: math.log(a*x + b) + c*x}),
    ("r = math.sqrt(x ** 2 + y ** 2)", lambda: {"inputs": [(3, 4), (5, 12), (1, 1)], "func": lambda x, y: math.sqrt(x**2 + y**2)}),
    
    # 3. Summation, loops & comprehensions
    ("total = sum([i ** 2 for i in range(1, {n} + 1)])", lambda n: {"inputs": [], "func": lambda: sum([i**2 for i in range(1, n + 1)])}),
    ("res = sum([1 / (i ** 2) for i in range(1, {n} + 1)])", lambda n: {"inputs": [], "func": lambda: sum([1 / (i**2) for i in range(1, n + 1)])}),
    ("s = sum([x ** i / math.factorial(i) for i in range({n})])", lambda n: {"inputs": [0.5, 1.0, 2.0], "func": lambda x: sum([x**i / math.factorial(i) for i in range(n)])}),
    
    # 4. Function definitions
    ("def solve(a, b): return math.sqrt(a ** 2 + b ** 2)", lambda: {"inputs": [(3, 4), (6, 8)], "func": lambda a, b: math.sqrt(a**2 + b**2)}),
    ("def poly(x): return {a} * x ** 2 + {b} * x + {c}", lambda a, b, c: {"inputs": [1, 2, 3], "func": lambda x: a*x**2 + b*x + c}),
    ("def area_circle(r): return math.pi * r ** 2", lambda: {"inputs": [1, 2, 3.5], "func": lambda r: math.pi * r**2}),
    ("def sigmoid(x): return 1 / (1 + math.exp(-x))", lambda: {"inputs": [0, 1, -2], "func": lambda x: 1 / (1 + math.exp(-x))}),
    ("def relu(x): return max(0, x)", lambda: {"inputs": [-3, 0, 5], "func": lambda x: max(0, x)}),

    # 5. Conditionals
    ("y = {a} * x if x > 0 else -{b} * x", lambda a, b: {"inputs": [2, -3, 0], "func": lambda x: a*x if x > 0 else -b*x}),
    ("val = math.log(x) if x > 0 else 0", lambda: {"inputs": [1, 2.718, -1], "func": lambda x: math.log(x) if x > 0 else 0}),
    ("sign = 1 if x > 0 else (-1 if x < 0 else 0)", lambda: {"inputs": [5, -4, 0], "func": lambda x: 1 if x > 0 else (-1 if x < 0 else 0)}),

    # 6. NumPy vector operations
    ("res = np.mean(np.array([{a}, {b}, {c}, {d}]))", lambda a, b, c, d: {"inputs": [], "func": lambda: np.mean(np.array([a, b, c, d]))}),
    ("norm = np.linalg.norm(np.array([{a}, {b}, {c}]))", lambda a, b, c: {"inputs": [], "func": lambda: np.linalg.norm(np.array([a, b, c]))}),
    ("dot = np.dot(np.array([{a}, {b}]), np.array([{c}, {d}]))", lambda a, b, c, d: {"inputs": [], "func": lambda: np.dot(np.array([a, b]), np.array([c, d]))}),
    ("std = np.std(np.array([{a}, {b}, {c}, {d}]))", lambda a, b, c, d: {"inputs": [], "func": lambda: np.std(np.array([a, b, c, d]))})
]

def generate_expression_instance(seed_val):
    rng = random.Random(seed_val)
    template_str, func_builder = rng.choice(CODE_TEMPLATES)
    params = {
        "a": rng.randint(1, 9),
        "b": rng.randint(1, 9),
        "c": rng.randint(1, 9),
        "d": rng.randint(1, 9),
        "n": rng.randint(3, 8)
    }
    keys_needed = [k[1:-1] for k in re.findall(r"\{[a-z]\}", template_str)]
    kwargs = {k: params[k] for k in keys_needed}
    code_str = template_str.format(**kwargs)
    return code_str

def render_handwritten_style_code(code_str, seed_val, width=512, height=64):
    rng = random.Random(seed_val)
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    
    font_size = rng.randint(16, 20)
    font_candidates = ["segoesc.ttf", "comic.ttf", "consola.ttf", "calibri.ttf", "arial.ttf"]
    font = None
    for fc in font_candidates:
        try:
            font = ImageFont.truetype(fc, font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    start_x = rng.randint(10, 25)
    start_y = rng.randint(15, 25)
    
    curr_x = start_x
    for char in code_str:
        char_y = start_y + rng.randint(-2, 2)
        draw.text((curr_x, char_y), char, fill=rng.randint(0, 45), font=font)
        bbox = draw.textbbox((0, 0), char, font=font)
        char_w = max(bbox[2] - bbox[0], 8)
        curr_x += char_w + rng.randint(0, 2)
        if curr_x > width - 20:
            break

    arr = np.array(img, dtype=np.float32)
    np_rng = np.random.RandomState(seed_val)
    noise = np_rng.normal(0, 4, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

# 2. Unified Math + Python Code Vocabulary
print("\n[1/5] Constructing Unified Math & Programming Vocabulary...")
MATH_TOKENS = [
    "\\frac", "\\sqrt", "\\alpha", "\\beta", "\\gamma", "\\theta", "\\pi", "\\sum", "\\int",
    "\\lim", "\\infty", "\\pm", "\\times", "\\div", "\\neq", "\\leq", "\\geq", "\\approx",
    "\\prime", "\\partial", "\\sin", "\\cos", "\\tan", "\\log", "\\ln", "\\angle", "\\circ",
    "\\triangle", "\\cdots", "\\ldots", "\\Delta", "\\lambda", "\\mu", "\\sigma", "\\omega"
]

CODE_TOKENS = [
    "def", "return", "lambda", "for", "in", "range", "if", "else", "sum", "math", "np",
    "math.sqrt", "math.sin", "math.cos", "math.exp", "math.log", "math.pi", "math.factorial",
    "np.array", "np.mean", "np.std", "np.dot", "np.linalg.norm", "max", "min", "res", "ans",
    "total", "norm", "poly", "solve", "area_circle", "sigmoid", "relu", "sign", "dot", "std",
    "**", "//", "==", "!=", "<=", ">=", "+=", "-=", "*=", "->",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
    "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "+", "-", "*", "/", "=", "(", ")", "[", "]", "{", "}", ":", ",", ".", ";", "<", ">", "_", "^"
]

ALL_UNIQUE = sorted(set(MATH_TOKENS + CODE_TOKENS))
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
token_to_id = {"<PAD>": PAD_ID, "<UNK>": UNK_ID, "<BOS>": BOS_ID, "<EOS>": EOS_ID}
for idx, tok in enumerate(ALL_UNIQUE):
    token_to_id[tok] = idx + 4
id_to_token = {v: k for k, v in token_to_id.items()}
VOCAB_SIZE = len(token_to_id)
print(f"  Unified Vocabulary Size: {VOCAB_SIZE} unique symbols/tokens")

# 3. In-Memory PyTorch Dataset
IMG_HEIGHT, IMG_WIDTH = 64, 512
MAX_LEN = 80

def tokenize_code(code_str):
    pattern = r"(math\.\w+|np\.\w+(?:\.\w+)?|\*\*|//|==|!=|<=|>=|\+=|-=|\*=|->|\w+|[^\s\w])"
    return re.findall(pattern, code_str)

def process_img(pil_img):
    img = pil_img.convert("L").resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img, dtype=np.float32)
    if arr.mean() > 127:
        arr = 255.0 - arr
    arr = arr / 255.0
    return torch.tensor(arr).unsqueeze(0)

class InMemCodeDataset(Dataset):
    def __init__(self, size, base_seed):
        self.size = size
        self.base_seed = base_seed
        self.code_strings = [generate_expression_instance(base_seed + i) for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        code_str = self.code_strings[idx]
        seed_val = self.base_seed + idx
        img_pil = render_handwritten_style_code(code_str, seed_val)
        img_t = process_img(img_pil)
        
        tokens = tokenize_code(code_str)
        ids = [token_to_id.get(t, UNK_ID) for t in tokens][:MAX_LEN - 2]
        
        dec_in = [BOS_ID] + ids
        dec_out = ids + [EOS_ID]
        
        dec_in += [PAD_ID] * (MAX_LEN - len(dec_in))
        dec_out += [PAD_ID] * (MAX_LEN - len(dec_out))
        
        return (
            img_t,
            torch.tensor(dec_in, dtype=torch.long),
            torch.tensor(dec_out, dtype=torch.long),
            code_str
        )

train_dataset = InMemCodeDataset(size=3000, base_seed=1000)
val_dataset   = InMemCodeDataset(size=400, base_seed=5000)
test_dataset  = InMemCodeDataset(size=300, base_seed=9000)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

print(f"  In-Memory Dataset -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# 4. Custom 2D-Coverage Architecture
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

class CodeHMERModel(nn.Module):
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

    def forward(self, images, dec_in):
        enc_features = self.encode(images)
        emb = self.embedding(dec_in)
        dec_out, _ = self.decoder_lstm(emb)
        context, _ = self.attention(dec_out, enc_features)
        combined = torch.cat([dec_out, context], dim=-1)
        logits = self.classifier(combined)
        return logits

# 5. Train Custom Code-HMER Model
print("\n[2/5] Training Custom 2D-Coverage Code Model on GPU...")
custom_code_model = CodeHMERModel().to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.05)
optimizer = torch.optim.AdamW(custom_code_model.parameters(), lr=5e-4, weight_decay=1e-4)
scaler = torch.amp.GradScaler('cuda' if DEVICE.type == 'cuda' else 'cpu')

EPOCHS = 10
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    custom_code_model.train()
    total_train_loss = 0.0
    t0 = time.time()
    
    for imgs, dec_in, dec_out, _ in train_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        dec_in = dec_in.to(DEVICE, non_blocking=True)
        dec_out = dec_out.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda' if DEVICE.type == 'cuda' else 'cpu'):
            logits = custom_code_model(imgs, dec_in)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), dec_out.reshape(-1))
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(custom_code_model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        total_train_loss += loss.item()

    # Validation
    custom_code_model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for imgs, dec_in, dec_out, _ in val_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            dec_in = dec_in.to(DEVICE, non_blocking=True)
            dec_out = dec_out.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda' if DEVICE.type == 'cuda' else 'cpu'):
                logits = custom_code_model(imgs, dec_in)
                v_loss = criterion(logits.reshape(-1, VOCAB_SIZE), dec_out.reshape(-1))
            total_val_loss += v_loss.item()
            
    avg_train = total_train_loss / len(train_loader)
    avg_val = total_val_loss / len(val_loader)
    print(f"  Custom Model Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Time: {time.time()-t0:.1f}s")

# 6. Fine-Tune Pretrained TrOCR Model on Code Dataset
print("\n[3/5] Fine-Tuning Pretrained TrOCR on Code Dataset...")
model_name = "microsoft/trocr-small-stage1"
trocr_processor = TrOCRProcessor.from_pretrained(model_name)
trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name).to(DEVICE)
trocr_model.config.decoder_start_token_id = trocr_processor.tokenizer.cls_token_id
trocr_model.config.pad_token_id = trocr_processor.tokenizer.pad_token_id

for p in trocr_model.encoder.parameters():
    p.requires_grad = False

trocr_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class InMemTrOCRDataset(Dataset):
    def __init__(self, code_dataset, tokenizer, max_len=64):
        self.code_ds = code_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.code_ds)

    def __getitem__(self, idx):
        code_str = self.code_ds.code_strings[idx]
        seed_val = self.code_ds.base_seed + idx
        img_pil = render_handwritten_style_code(code_str, seed_val).convert("RGB")
        img_tensor = trocr_transform(img_pil)
        
        lbl = self.tokenizer(code_str, padding="max_length", max_length=self.max_len, truncation=True, return_tensors="pt").input_ids.squeeze(0)
        lbl[lbl == self.tokenizer.pad_token_id] = -100
        return img_tensor, lbl

train_trocr_loader = DataLoader(InMemTrOCRDataset(train_dataset, trocr_processor.tokenizer), batch_size=16, shuffle=True)
trocr_opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, trocr_model.parameters()), lr=1e-4, weight_decay=1e-4)

for ep in range(1, 3):
    trocr_model.train()
    t_loss = 0.0
    t0 = time.time()
    for p_vals, lbls in train_trocr_loader:
        p_vals = p_vals.to(DEVICE, non_blocking=True)
        lbls = lbls.to(DEVICE, non_blocking=True)
        trocr_opt.zero_grad()
        with torch.amp.autocast('cuda' if DEVICE.type == 'cuda' else 'cpu'):
            out = trocr_model(pixel_values=p_vals, labels=lbls)
            loss = out.loss
        scaler.scale(loss).backward()
        scaler.unscale_(trocr_opt)
        torch.nn.utils.clip_grad_norm_(trocr_model.parameters(), 1.0)
        scaler.step(trocr_opt)
        scaler.update()
        t_loss += loss.item()
    print(f"  TrOCR Code Epoch {ep}/2 | Loss: {t_loss/len(train_trocr_loader):.4f} | Time: {time.time()-t0:.1f}s")

# 7. Code Execution & Verification Engine
print("\n[4/5] Initializing Python Code Execution & Verification Engine...")

def execute_code_safe(code_str):
    """
    Executes recognized Python code safely, checks syntax,
    and returns validity and execution status.
    """
    try:
        ast.parse(code_str)
    except SyntaxError as e:
        return {"valid_syntax": False, "exec_success": False, "result": None, "error": f"SyntaxError"}

    sandbox_globals = {
        "math": math,
        "np": np,
        "sum": sum,
        "max": max,
        "min": min,
        "abs": abs,
        "range": range,
        "x": 2,
        "y": 3,
        "a": 2,
        "b": 3,
        "c": 4,
        "d": 1,
        "n": 5
    }
    local_vars = {}
    
    try:
        exec(code_str, sandbox_globals, local_vars)
        output_keys = ["res", "y", "ans", "total", "norm", "dot", "std", "s", "val", "sign", "f", "solve", "poly", "area_circle", "sigmoid", "relu"]
        result = None
        for k in output_keys:
            if k in local_vars:
                res_val = local_vars[k]
                if callable(res_val):
                    try:
                        result = res_val(2)
                    except Exception:
                        result = "function_defined"
                else:
                    result = res_val
                break
        return {"valid_syntax": True, "exec_success": True, "result": result, "error": None}
    except Exception:
        return {"valid_syntax": True, "exec_success": False, "result": None, "error": "RuntimeError"}

# 8. Complete Recognition + Execution Evaluation
print(f"\n[5/5] Evaluating Both Models on {len(test_dataset)} Unseen Test Expressions...")
custom_code_model.eval()
trocr_model.eval()

def decode_custom(model, images_pil):
    tensors = torch.stack([process_img(img) for img in images_pil]).to(DEVICE)
    with torch.no_grad():
        enc_f = model.encode(tensors)
        B_size = enc_f.shape[0]
        sp_pts = enc_f.shape[1]
        c_toks = torch.full((B_size, 1), BOS_ID, dtype=torch.long, device=DEVICE)
        h = torch.zeros(1, B_size, 384, device=DEVICE)
        c = torch.zeros(1, B_size, 384, device=DEVICE)
        cov = torch.zeros(B_size, sp_pts, 1, device=DEVICE)
        gen = [[] for _ in range(B_size)]
        active = np.ones(B_size, dtype=bool)

        for _ in range(MAX_LEN):
            emb = model.embedding(c_toks)
            dec_out, (h, c) = model.decoder_lstm(emb, (h, c))
            ctx, att_w = model.attention(dec_out, enc_f, coverage=cov)
            cov = cov + att_w.transpose(1, 2)
            comb = torch.cat([dec_out, ctx], dim=-1)
            logits = model.classifier(comb)
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
    return [" ".join(g) for g in gen]

custom_preds = []
trocr_preds = []
ground_truths = []

batch_size = 16
num_batches = math.ceil(len(test_dataset) / batch_size)

for b in range(num_batches):
    indices = list(range(b * batch_size, min((b + 1) * batch_size, len(test_dataset))))
    batch_codes = [test_dataset.code_strings[i] for i in indices]
    images_pil = [render_handwritten_style_code(c, test_dataset.base_seed + i) for i, c in zip(indices, batch_codes)]
    ground_truths.extend(batch_codes)

    # 1. Custom Prediction
    custom_pred_raw = decode_custom(custom_code_model, images_pil)
    for pr in custom_pred_raw:
        clean_code = pr.replace(" . ", ".").replace(" ( ", "(").replace(" )", ")").replace(" [ ", "[").replace(" ]", "]").replace(" :", ":").replace(" , ", ", ")
        clean_code = clean_code.replace(" * * ", "**").replace(" / / ", "//").replace(" = = ", "==").replace(" ! = ", "!=").replace(" + = ", "+=")
        custom_preds.append(clean_code)

    # 2. TrOCR Prediction
    pixel_values = torch.stack([trocr_transform(img.convert("RGB")) for img in images_pil]).to(DEVICE)
    with torch.no_grad():
        gen_tr = trocr_model.generate(pixel_values, max_new_tokens=64)
    trocr_preds.extend(trocr_processor.batch_decode(gen_tr, skip_special_tokens=True))

# Compute Recognition & Execution Metrics
def evaluate_all(preds, truths):
    exact_match = 0
    valid_syntax_count = 0
    exec_success_count = 0
    functional_match_count = 0
    total_sim = 0.0
    N = len(preds)

    for p, g in zip(preds, truths):
        if p.strip() == g.strip():
            exact_match += 1
        
        p_toks, g_toks = p.split(), g.split()
        dp = list(range(len(g_toks) + 1))
        for i in range(1, len(p_toks) + 1):
            ndp = [i] + [0] * len(g_toks)
            for j in range(1, len(g_toks) + 1):
                if p_toks[i - 1] == g_toks[j - 1]:
                    ndp[j] = dp[j - 1]
                else:
                    ndp[j] = 1 + min(dp[j], ndp[j - 1], dp[j - 1])
            dp = ndp
        dist = dp[-1]
        sim = max(0.0, 1.0 - dist / max(len(p_toks), len(g_toks), 1))
        total_sim += sim

        # Execution Evaluation
        p_exec = execute_code_safe(p)
        g_exec = execute_code_safe(g)

        if p_exec["valid_syntax"]:
            valid_syntax_count += 1
        if p_exec["exec_success"]:
            exec_success_count += 1
            if g_exec["exec_success"] and p_exec["result"] is not None:
                try:
                    if isinstance(p_exec["result"], (int, float, np.number)) and isinstance(g_exec["result"], (int, float, np.number)):
                        if math.isclose(float(p_exec["result"]), float(g_exec["result"]), rel_tol=1e-3, abs_tol=1e-3):
                            functional_match_count += 1
                    elif str(p_exec["result"]) == str(g_exec["result"]):
                        functional_match_count += 1
                except Exception:
                    pass

    return {
        "exact_match": (exact_match / N) * 100,
        "similarity": (total_sim / N) * 100,
        "syntax_validity": (valid_syntax_count / N) * 100,
        "exec_success": (exec_success_count / N) * 100,
        "functional_correctness": (functional_match_count / N) * 100
    }

custom_res = evaluate_all(custom_preds, ground_truths)
trocr_res = evaluate_all(trocr_preds, ground_truths)

print("\n" + "=" * 90)
print("FINAL BENCHMARK: HANDWRITTEN PROGRAMMING RECOGNITION & EXECUTION ACCURACY")
print("=" * 90)
print(f"{'Evaluation Metric / Dimension':<35} | {'Custom 2D-CNN Model':<22} | {'Fine-Tuned TrOCR':<20}")
print("-" * 90)
print(f"{'Exact Code String Match (%)':<35} | {custom_res['exact_match']:<22.2f} | {trocr_res['exact_match']:<20.2f}")
print(f"{'Token-Level Similarity (%)':<35} | {custom_res['similarity']:<22.2f} | {trocr_res['similarity']:<20.2f}")
print(f"{'Syntax Validity Rate (AST Valid %)':<35} | {custom_res['syntax_validity']:<22.2f} | {trocr_res['syntax_validity']:<20.2f}")
print(f"{'Execution Success Rate (%)':<35} | {custom_res['exec_success']:<22.2f} | {trocr_res['exec_success']:<20.2f}")
print(f"{'Functional Output Correctness (%)':<35} | {custom_res['functional_correctness']:<22.2f} | {trocr_res['functional_correctness']:<20.2f}")
print("=" * 90)

print("\n--- Live Code Execution Sample Inspection ---")
for i in range(min(5, len(ground_truths))):
    p_code = custom_preds[i]
    g_code = ground_truths[i]
    exec_res = execute_code_safe(p_code)
    print(f"\n[Test Code Sample #{i+1}]")
    print(f"  Ground Truth Code : {g_code}")
    print(f"  Recognized Code   : {p_code}")
    print(f"  Syntax Valid?     : {'YES [OK]' if exec_res['valid_syntax'] else 'NO [X]'}")
    print(f"  Executed Result   : {exec_res['result']}")

print("\n" + "=" * 90)
print("PROGRAMMING EXPRESSION RECOGNITION & EXECUTION COMPLETE!")
print("=" * 90)
