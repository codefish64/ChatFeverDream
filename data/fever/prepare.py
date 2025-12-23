import re
import numpy as np
import sentencepiece as spm
import os

# Step 1: read raw
text = open('data/fever/raw.txt', 'r', encoding='utf-8', errors='ignore').read()

# Step 2: minimal cleaning
text = re.sub(r"http\S+", "", text)         # remove URLs
text = re.sub(r"\n{3,}", "\n\n", text)     # collapse blank lines

# Step 3: train tokenizer (if not already done)
if not os.path.exists('data/fever/fever.model'):
    spm.SentencePieceTrainer.train(
        input='data/fever/raw.txt',
        model_prefix='data/fever/fever',
        vocab_size=8000,
        character_coverage=1.0,
        model_type='bpe'
    )

# Step 4: encode dataset
sp = spm.SentencePieceProcessor()
sp.load('data/fever/fever.model')

ids = sp.encode(text)
split = int(0.9 * len(ids))
train_ids = np.array(ids[:split], dtype=np.uint16)
val_ids   = np.array(ids[split:], dtype=np.uint16)

train_ids.tofile('data/fever/train.bin')
val_ids.tofile('data/fever/val.bin')

print("Finished preparing data! train.bin & val.bin created.")

