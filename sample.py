"""
FeverdreamGPT Sampling Script
Generates text from a trained FeverdreamGPT model
"""
import os
import pickle
import torch
import sentencepiece as spm
from model import GPTConfig, GPT
from contextlib import nullcontext

# -------------------- config --------------------
out_dir = "out-fever"        # directory where checkpoint is saved
num_samples = 5              # number of samples to generate
max_new_tokens = 150         # number of tokens per sample
temperature = 0.9            # randomness control
top_k = 200                  # top-k filtering
device = "cpu"               # force CPU for safety on M1
seed = 1337
start_prompt = "chat listen"           # initial text
compile_model = False         # torch.compile optional
# ------------------------------------------------

# Seed
torch.manual_seed(seed)

# Autocast context (mostly for float16 compatibility if needed)
ctx = nullcontext()  # CPU-only, no AMP required

# ---------------- load model -------------------
ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location="cpu")
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)

# fix any legacy prefixes in state dict
state_dict = checkpoint['model']
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()
model.to(device)
if compile_model:
    model = torch.compile(model)

# ----------------- load tokenizer -----------------
meta_path = os.path.join("data", checkpoint['config']['dataset'], "meta.pkl")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)

if meta.get("tokenizer") == "sentencepiece":
    sp = spm.SentencePieceProcessor()
    sp_model_path = os.path.join(os.path.dirname(meta_path), meta["tokenizer_model"])
    sp.load(sp_model_path)
    encode = lambda s: sp.encode(s, out_type=int)
    decode = lambda l: sp.decode(l)
else:
    # fallback to char-level (unlikely)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

# ----------------- prompt -----------------
if start_prompt.startswith("FILE:"):
    with open(start_prompt[5:], "r", encoding="utf-8") as f:
        start_prompt = f.read()

start_ids = encode(start_prompt)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# ----------------- gen -----------------
with torch.no_grad():
    with ctx:
        for i in range(num_samples):
            y = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
            print(decode(y[0].tolist()))
            print("---------------")
