"""
FeverdreamGPT Chat Interface
"""
import os
import torch
import pickle
import sentencepiece as spm
import gradio as gr
from model import GPT, GPTConfig

# ---------------- config ----------------
checkpoint_dir = "out-fever"     # ckpt.pt folder
dataset_dir = "data/fever"       # meta.pkl and tokenizer folder
device = "cpu"                   # only use cpu
max_new_tokens = 150
num_samples_per_input = 1
# ---------------------------------------

#load model
ckpt_path = os.path.join(checkpoint_dir, "ckpt.pt")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

checkpoint = torch.load(ckpt_path, map_location=device)
model_conf = GPTConfig(**checkpoint["model_args"])
model = GPT(model_conf)

state_dict = checkpoint["model"]
prefix = "_orig_mod."
for k in list(state_dict.keys()):
    if k.startswith(prefix):
        state_dict[k[len(prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()
model.to(device)

#load tokenizer
meta_path = os.path.join(dataset_dir, "meta.pkl")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)

if meta.get("tokenizer") == "sentencepiece":
    sp = spm.SentencePieceProcessor()
    sp_model_path = os.path.join(dataset_dir, meta["tokenizer_model"])
    sp.load(sp_model_path)
    encode = lambda s: sp.encode(s, out_type=int)
    decode = lambda l: sp.decode(l)
else:
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

#chat function
def chat(input_text, insanity, focus, history=None):
    if history is None:
        history = []
    if not input_text.strip():
        input_text = "Chat listen "  # default prompt


    history.append({"role": "user", "content": input_text})

    x = torch.tensor(encode(input_text), dtype=torch.long, device=device)[None, ...]
    top_k = int(focus)
    with torch.no_grad():
        for _ in range(num_samples_per_input):
            y = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=insanity,
                top_k=top_k
            )
            output_text = decode(y[0].tolist())
            history.append({"role": "assistant", "content": output_text})

    return history, history

#interface
iface = gr.Interface(
    fn=chat,
    inputs=[
        gr.Textbox(label="Your message"),
        gr.Slider(minimum=0.5, maximum=10, value=0.9, step=0.05, label="Insanity"),
        gr.Slider(minimum=20, maximum=200, value=200, step=5, label="Randomness< >Focus"),
        gr.State(value=[])
    ],
    outputs=[gr.Chatbot(label="FeverdreamGPT"), gr.State(value=[])],
    title="FeverdreamGPT",
    description="A micro AI trained on shitposts and copypastas. Hallucinates utter bullshit, may be explicit."
)

iface.launch()