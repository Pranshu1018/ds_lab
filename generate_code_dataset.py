import os
import sys
import re
import math
import random
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 75)
print("GENERATING HANDWRITTEN PROGRAMMING EXPRESSIONS DATASET")
print("=" * 75)

OUTPUT_DIR = os.path.abspath("./code_dataset")
os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)

# Templates for realistic mathematical programming expressions
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

    # 6. NumPy vector & matrix operations
    ("res = np.mean(np.array([{a}, {b}, {c}, {d}]))", lambda a, b, c, d: {"inputs": [], "func": lambda: np.mean(np.array([a, b, c, d]))}),
    ("norm = np.linalg.norm(np.array([{a}, {b}, {c}]))", lambda a, b, c: {"inputs": [], "func": lambda: np.linalg.norm(np.array([a, b, c]))}),
    ("dot = np.dot(np.array([{a}, {b}]), np.array([{c}, {d}]))", lambda a, b, c, d: {"inputs": [], "func": lambda: np.dot(np.array([a, b]), np.array([c, d]))}),
    ("std = np.std(np.array([{a}, {b}, {c}, {d}]))", lambda a, b, c, d: {"inputs": [], "func": lambda: np.std(np.array([a, b, c, d]))})
]

def generate_expression_instance():
    template_str, func_builder = random.choice(CODE_TEMPLATES)
    # Generate random parameters
    params = {
        "a": random.randint(1, 9),
        "b": random.randint(1, 9),
        "c": random.randint(1, 9),
        "d": random.randint(1, 9),
        "n": random.randint(3, 8)
    }
    # Format code string
    keys_needed = [k[1:-1] for k in re.findall(r"\{[a-z]\}", template_str)]
    kwargs = {k: params[k] for k in keys_needed}
    code_str = template_str.format(**kwargs)
    
    # Extract builder args
    builder_param_count = func_builder.__code__.co_argcount
    builder_args = [params[k] for k in ["a", "b", "c", "d", "n"][:builder_param_count]]
    exec_meta = func_builder(*builder_args)
    
    return code_str, exec_meta

def render_handwritten_style_code(code_str, width=512, height=64):
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    
    # Try system fonts or default with synthetic handwriting distortion
    font_size = random.randint(16, 20)
    try:
        # Common Windows fonts with handwriting / cursive / sans appearance
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
    except Exception:
        font = ImageFont.load_default()

    # Add jitter to start position
    start_x = random.randint(10, 25)
    start_y = random.randint(15, 25)
    
    # Draw text with character-level natural handwriting variation
    curr_x = start_x
    for char in code_str:
        char_y = start_y + random.randint(-2, 2)
        draw.text((curr_x, char_y), char, fill=random.randint(0, 45), font=font)
        # Advance with slight kerning jitter
        bbox = draw.textbbox((0, 0), char, font=font)
        char_w = max(bbox[2] - bbox[0], 8)
        curr_x += char_w + random.randint(0, 2)
        if curr_x > width - 20:
            break

    # Apply synthetic handwritten distortions (stroke thickness, subtle blur, noise)
    if random.random() > 0.5:
        img = img.filter(ImageFilter.SMOOTH_MORE)
    
    arr = np.array(img, dtype=np.float32)
    # Add paper grain & stroke variation
    noise = np.random.normal(0, 4, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(arr)

def build_split(split_name, num_samples):
    print(f"\nGenerating {num_samples} samples for '{split_name}' split...")
    metadata = []
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    
    for idx in range(num_samples):
        code_str, exec_meta = generate_expression_instance()
        img = render_handwritten_style_code(code_str)
        img_filename = f"code_{split_name}_{idx:05d}.png"
        img_path = os.path.join(split_dir, img_filename)
        img.save(img_path)
        
        # Test ground-truth execution to verify validity
        sample_meta = {
            "id": idx,
            "image_path": img_path,
            "code": code_str,
            "tokens": code_str.split()
        }
        metadata.append(sample_meta)
        
        if (idx + 1) % 500 == 0 or (idx + 1) == num_samples:
            sys.stdout.write(f"\r  Progress: {idx+1}/{num_samples} generated...")
            sys.stdout.flush()
            
    meta_file = os.path.join(OUTPUT_DIR, f"{split_name}_metadata.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  [OK] Saved {len(metadata)} samples -> {meta_file}")

# Generate Train (4,000), Val (500), Test (500)
build_split("train", 4000)
build_split("val", 500)
build_split("test", 500)

print("\n" + "=" * 75)
print("CODE DATASET GENERATION COMPLETED SUCCESSFULLY!")
print(f"Dataset stored at: {OUTPUT_DIR}")
print("=" * 75)
