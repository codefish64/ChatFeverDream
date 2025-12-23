# ChatFeverDream
ChatFeverDream - a delightfully deranged AI that is trained off of ~100mb of pure Reddit copypasta and shitpost extract.

## About:
- Nearly unintelligible
- Schizophrenic
- 13.7 million parameter (sounds like a lot but is not)
- Runs locally on apple silicon CPU
- Web UI built with gradio
- Adjustable insanity
- Adjustable focus and randomness

## Disclaimer
- if this thing figures out how to give you advice, DO NOT LISTEN TO IT
- May generate very explicit language, including slurs, etc. (its reddit what would you expect)
- educational purposes only or something

## Uses:
- nanoGPT / miniGPT (MIT License)
- PyTorch
- SentencePiece tokenizer
- Gradio web interface (Apache 2.0)

# Setup & Install
## Requirements
- Macos Apple Silicon (M-series)
- Python 3.10–3.12 (3.11 recommended)
- ~2–3 GB free RAM
- No GPU required cause im chill like that

## Install:
### clone the repo
```bash
git clone https://github.com/yourusername/ChatFeverDream.git
cd ChatFeverDream
```
### create ur venv
```bash
python3 -m venv feverllm
source feverllm/bin/activate
```
### upgrade ur pip
```bash
pip install --upgrade pip
```
### install requirments
```bash
pip install -r requirements.txt
```
### run it!
```bash
python /Users/Lazertron/nanoGPT/chatFeverDream.py
```
Go to this URL and it will be there running on your local host in port 7860!
http://127.0.0.1:7860

## Model
There is a pretrained model for you at out-fever/ckpt.pt. If you wanna retrain, there are resources. Figure it out idk

## Acknowledgements
**NanoGPT** https://github.com/karpathy/nanoGPT (MIT License)
**Gradio** https://github.com/gradio-app/gradio (Apache 2.0 License)

## uh yeah thats it have fun
