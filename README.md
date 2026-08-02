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

Needs Python 3.8+ and one library (numpy). Run every command **from inside
this folder**. On Mac/Linux type `python3` and `pip3` wherever you see
`python` and `pip`; on Windows, if `python` opens the Microsoft Store, use
`py` instead.

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
You can press ctrl-c any time — it saves its progress before exiting (and
auto-saves every 500 steps during the run).

### First time on Windows?

1. Install Python from https://www.python.org/downloads — and **check the
   box "Add python.exe to PATH"** on the installer's first screen.
2. Get this code: green **Code** button above → **Download ZIP** → right-click
   → Extract All.
3. Open a terminal in the extracted folder: in File Explorer, click the
   address bar, type `cmd`, press Enter.
4. Run the Quick start commands. (Windows paths use `\`, e.g.
   `checkpoints\shakespeare.npz` — both usually work here.)

### First time on a Mac?

1. Open Terminal (Cmd+Space, type "Terminal").
2. Check `python3 --version` — if it's missing or older than 3.8, install
   from https://www.python.org/downloads.
3. Download the ZIP as above, unzip, then `cd` into it (e.g.
   `cd ~/Downloads/AI-main`) and run the Quick start with `python3`/`pip3`.

Training makes your CPU work hard — fans are normal, laptops should be
plugged in, and the computer must not go to sleep (screen off is fine).

## Teach it to code

```bash
python train.py --dataset code          # real Python stdlib source, ~10-15 min
python generate.py --model checkpoints/code.npz --prompt "def " --tokens 400
```

### 27 languages

`download_data.py` can fetch real source code in any of these languages —
list them any time with `python download_data.py languages`:

```
bash        c           cpp         csharp      gdscript    glsl
go          hlsl        java        javascript  julia       kotlin
lua         matlab      micropython perl        php         powershell
python      r           ruby        rust        scala       shell
swift       typescript  wgsl
```

Train on just one:

```bash
python download_data.py lang rust
python train.py --data data/lang_rust.txt --out checkpoints/rust
python generate.py --model checkpoints/rust.npz --prompt "fn "
```

Or build a **polyglot** brain that learns several languages at once — your
prompt's opening characters steer which language it continues in (`def ` →
Python, `func ` → Go, `fn ` → Rust):

```bash
python download_data.py polyglot python rust go lua csharp   # pick any subset
# or every language at once:
python download_data.py polyglot

python train.py --data data/polyglot_python_rust_go_lua_csharp.txt \
                 --layers 6 --emb 192 --steps 30000 --out checkpoints/poly

python generate.py --model checkpoints/poly.npz --prompt "public class "
```

More languages sharing one small brain means each gets less "room," so a
polyglot model benefits even more from the bigger `--layers`/`--emb`/
`--steps` settings under "Make it smarter" below.

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

Two things to know about `--resume`: the size flags (`--layers`, `--emb`,
`--context`) are ignored — a saved brain keeps its shape forever. And a
brain's alphabet is fixed the first time it's trained, so if you resume on
text full of characters it has never seen, those get skipped (the trainer
warns you). For very different text, train a fresh brain on a combined file
instead.

## Setting boundaries (words it's not allowed to say)

The model has no idea any word is "bad" — it can only say what was in its
training text, and if you feed it text with cursing in it, cursing can come
back out. Real AI systems handle this with two layers, and so does this one:

1. **Curate the diet.** The first boundary is choosing training data. The
   built-in datasets are tame; if you don't want a kind of language coming
   out, don't train it on text that contains it.
2. **The boundary file.** `banned_words.txt` lists words the model is never
   allowed to write — one per line, edit it freely. During generation, any
   character that would complete a banned word is vetoed *before it's shown*
   and the model is forced to pick different words. The block is absolute:
   banned words cannot appear, in any letter case.

```bash
# on by default; point at your own list, or '' to turn off
python generate.py --model checkpoints/shakespeare.npz --banned my_rules.txt
```

One quirk to know: banning a word also blocks longer words that *start* with
it (banning "hell" also blocks "hello"), so ban the most specific form you
can. This filter is a miniature of the "guardrails" real AI products use —
and the reason big labs ALSO train models to refuse (rather than only
filtering) is exactly the weakness you can spot here: a word filter can't
judge meaning, only spelling.

## What's in the files

| File | What it is |
|------|------------|
| `model.py` | The GPT itself: attention, transformer blocks, backprop, Adam, checkpoints |
| `train.py` | The training loop: data → loss → gradients → nudge, repeat |
| `generate.py` | Load a trained brain and generate from a prompt (or `--interactive`) |
| `download_data.py` | Internet access: fetches training data from the web |
| `banned_words.txt` | The boundary: words it is never allowed to say |
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
