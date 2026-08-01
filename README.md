# AI — a real GPT built from scratch

A miniature GPT — the same architecture family as ChatGPT and Claude — written
from scratch in Python with numpy. No PyTorch, no TensorFlow, no APIs, no app
builders. Every forward pass and every gradient is code you can read.

It downloads its own training data from the internet, learns to write
Shakespeare-style English or Python-shaped code, saves its trained brain to a
file, and then generates text live from any prompt you give it.

**Upgrades over a basic from-scratch model:**

- **Smarter** — real multi-head self-attention (the "Attention Is All You
  Need" transformer, the actual breakthrough behind modern AI) instead of a
  fixed-window MLP. Stacked blocks, residual connections, layer norm: a true
  baby GPT.
- **Faster** — the Adam optimizer with warmup + cosine learning-rate schedule
  (trains many times faster than plain gradient descent), float32 math, and
  **checkpoints**: train once, generate forever. No more relearning every run.
- **It can code** — train it on ~500 KB of real Python from the official
  standard library and it learns to write Python-shaped code: `def`, colons,
  indentation, docstrings.
- **Internet access** — `download_data.py` fetches its training data from the
  web: Shakespeare, real Python source, or **any text URL you point it at**.

## Quick start

```bash
pip install numpy

# 1. prove the math is correct (30 seconds)
python test_model.py

# 2. train it on Shakespeare, downloaded live from the internet (~10-15 min)
python train.py --dataset shakespeare

# 3. talk to it
python generate.py --model checkpoints/shakespeare.npz --prompt "ROMEO:" 
python generate.py --model checkpoints/shakespeare.npz --interactive
```

While training you'll watch the loss fall and see live samples evolve from
random noise → words → sentences. That's learning happening in front of you.
You can press ctrl-c any time; a checkpoint is saved every 500 steps.

## Teach it to code

```bash
python train.py --dataset code          # real Python stdlib source, ~10-15 min
python generate.py --model checkpoints/code.npz --prompt "def " --tokens 400
```

## Train it on ANYTHING

Any text file, or any text URL on the internet:

```bash
# a whole free book from Project Gutenberg
python download_data.py url https://www.gutenberg.org/files/11/11-0.txt alice
python train.py --data data/alice.txt

# or your own writing
python train.py --data my_stuff.txt
```

## Make it smarter

Every knob is a flag. Bigger + longer = smarter (and slower):

```bash
python train.py --dataset shakespeare --layers 4 --emb 128 --context 128 --steps 8000
```

Continue training an existing brain instead of starting over:

```bash
python train.py --dataset shakespeare --resume checkpoints/shakespeare.npz --steps 3000
```

## What's in the files

| File | What it is |
|------|------------|
| `model.py` | The GPT itself: attention, transformer blocks, backprop, Adam, checkpoints |
| `train.py` | The training loop: data → loss → gradients → nudge, repeat |
| `generate.py` | Load a trained brain and generate from a prompt (or `--interactive`) |
| `download_data.py` | Internet access: fetches training data from the web |
| `tokenizer.py` | Text ↔ numbers (character-level) |
| `test_model.py` | Proof of correctness: checks backprop against brute-force calculus |

## Honest limits (read this)

This is the **real GPT recipe at 1/1,000,000th scale** (~350 thousand
parameters vs hundreds of billions; one small dataset vs most of the
internet). What that means:

- It genuinely **learns** — spelling, style, structure — from raw text, by
  itself. Nothing is hardcoded.
- The code model writes **code-shaped text**: it discovers Python's look
  (indentation, `def`, `self.`, docstrings) which is remarkable for something
  this small — but its programs won't actually run. Writing *working* code
  requires billions of parameters and vastly more data/compute.
- It can't chat or answer questions. It has one skill: continue the text.
  (Chat AIs are next-token predictors too — just huge ones, further trained
  to prefer helpful continuations.)
- "Internet access" here means it **learns from** the web (downloads its own
  training data). Live browsing-while-answering is a tool-use loop wrapped
  around a huge model — the loop is easy, the huge model is the hard part.

The point: after reading these ~700 lines, there is no remaining mystery in
how GPT-class AI works. The rest is scale.

## How it learns (one paragraph)

The model reads 64 characters and predicts a probability for the next one, at
every position at once. The loss measures how wrong those predictions are.
Backpropagation (the chain rule from calculus, written out by hand in
`model.py`) computes how every one of the ~350,000 weights should change to
reduce the loss, and Adam nudges them. Repeat a few thousand times and the
weights come to encode spelling, words, and style — knowledge nobody
programmed in. Generation is just: predict → sample a character → append →
repeat.
