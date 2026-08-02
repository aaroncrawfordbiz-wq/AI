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

## A text library, not just one play

```bash
python download_data.py library      # 19 full public-domain novels, ~18 MB
python train.py --data data/library.txt --layers 6 --emb 256 --heads 8 --context 128 --steps 40000
```

Too much text to memorize forces the model to actually learn language
patterns instead of parroting pages — the "student vs. parrot" difference.

## Everything at once — text library + all 27 languages, run for a full day

```bash
python download_data.py everything   # library + every language, ~20 MB

python train.py --data data/everything.txt \
                 --layers 7 --emb 320 --heads 8 --context 160 \
                 --hours 24 \
                 --out checkpoints/big
```

That's roughly **2x the parameters** of the library recipe above (~8.8
million vs ~4.8 million), and `--hours 24` means it trains for a full day
and stops — no step-count guessing. It checkpoints every 500 steps AND every
~10 minutes either way, so it's always safe to interrupt, and the printed
`steps/s` adapts to however fast YOUR PC actually runs it — the flag does
the math for you no matter the hardware.

After a day on this much varied text, expect a real step up: longer
coherent stretches, better grammar, and it switches between prose and any
of the 27 languages depending on how you start the prompt. Still the same
honest ceiling as ever, just further along it: genuinely smarter at
*sounding like* English and *sounding like* code, not a program that
executes or a mind that reasons — that gap is closed by scale (GPU
datacenters, not more hours on one CPU), not by dataset size alone.

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

## Real math (a calculator tool, not the neural net guessing)

The trained model cannot do arithmetic — it never learned numbers, only
which digits tend to follow others in text. `calculator.py` is a real,
exact calculator (Python's own math, safely sandboxed to plain `+ - * / **`
— no code execution possible) that answers instead of the model whenever it
detects a math question:

```
python generate.py --model checkpoints/shakespeare.npz --interactive
you> what is 47 * 83
(calculator, not the model): 47 * 83 = 3901
```

This is the real version of "can it do math" — not training it harder, but
giving it an actual tool and having plain code decide when to use it. This
is also literally how real AI products handle math.

## Business/marketing text (style only)

```bash
python download_data.py business   # My Life and Work (Ford) + The Art of War
python train.py --data data/business.txt
```

Same rule as everywhere else in this project: it learns the *tone and
vocabulary* of business writing, not real facts. It cannot tell you what a
market is doing, price anything, or run analysis — it has no access to
real data and no ability to reason about numbers (see the calculator above
for the one place real computation is possible). Training on this text
teaches it to *sound* like a business book, nothing more.

## Blender and Unreal Engine — real geometry, honestly built

`content_commands.py`, `blender_bridge.py`, and `unreal_bridge.py` let you
type `"red cube at 2 0 1 size 1.5"` and get a real object placed in a real
Blender scene or Unreal level.

**Read this before trying it:** the object is created by an exact,
ordinary keyword parser — **not** by the trained neural network. This is a
deliberate, honest design choice, not a shortcut: the README's own
generated samples throughout this project prove model.py produces
code-*shaped* text that doesn't run. Handing 3D creation to it would mean
wrong shapes in wrong places with no way to know until you looked. A real
parser is what actually, reliably works — the same "language model decides
intent, exact code executes it" split used by every real product that
combines AI with precise output. There is no version of "the AI designs
AAA content" that this scale of model can honestly do; that requires a
completely different kind of AI (3D-generation networks trained on 3D
datasets), not a bigger version of this text model.

```bash
# test without owning Blender/Unreal at all:
python blender_bridge.py --dry-run "red cube at -2 0 0; blue sphere at 0 0 0 size 1.2"
python unreal_bridge.py --dry-run "red cube at -200 0 0 size 100"
```

Each file's docstring has exact steps to run it for real inside Blender's
Scripting tab or Unreal's Python console. Optionally, `blender_bridge.py`
can ask a trained checkpoint for a short flavor-text caption of the scene
(`caption_scene()`) — clearly cosmetic, never parsed back into commands,
never controlling anything.

### Talk to Blender in plain English, via MCP

There IS a real way to say "build a small red house" in ordinary language
and have Blender build it correctly — just not by making the from-scratch
model in this repo smarter. The trick is [MCP](https://modelcontextprotocol.io)
(Model Context Protocol): it lets a genuinely capable AI — Claude Desktop,
Claude Code, or any MCP client — call Blender-building functions by name.
The big model's real job (understanding "a small red house" and breaking it
into shape commands) is something large language models are actually good
at; the exact building is still done by `content_commands.py`'s ordinary
parser, same as above. Three pieces, verified working together end to end
in this repo (including a live test over the real MCP protocol):

```
  You, in Claude Desktop         The big AI calls           Blender, actually
  or Claude Code, in     --->    build_shapes(...)   --->   building the
  plain English                  over MCP                   real object
                                       |
                          blender_addon_server.py's HTTP API
                          (a small server running INSIDE Blender)
```

Setup:

```bash
pip install "mcp>=1.6,<2"
```

1. Open Blender -> Scripting tab -> open `blender_addon_server.py` -> Run
   Script. Leave Blender open (see that file's docstring for details).
2. Add `blender_mcp_server.py` to your AI client's MCP config — for Claude
   Desktop, in `claude_desktop_config.json`:
   ```json
   { "mcpServers": { "blender": {
       "command": "python", "args": ["/full/path/to/blender_mcp_server.py"]
   } } }
   ```
3. Restart the client, then just ask in plain English: *"build a small red
   house out of cubes"* — the AI figures out the shape commands and calls
   the tool; Blender builds the real geometry.

Test the whole chain without Blender or an AI client at all:
```bash
python blender_addon_server.py --dry-run &
python blender_mcp_server.py --selftest
```

## What's in the files

| File | What it is |
|------|------------|
| `model.py` | The GPT itself: attention, transformer blocks, backprop, Adam, checkpoints |
| `train.py` | The training loop: data → loss → gradients → nudge, repeat |
| `generate.py` | Load a trained brain and generate from a prompt (or `--interactive`) |
| `download_data.py` | Internet access: fetches training data from the web |
| `banned_words.txt` | The boundary: words it is never allowed to say |
| `calculator.py` | A real calculator tool — used INSTEAD of the model for math |
| `content_commands.py` | Exact parser: text like "red cube at 0 0 0" -> a structured action |
| `blender_bridge.py` | Builds real objects in Blender from parsed commands |
| `unreal_bridge.py` | Spawns real actors in Unreal Engine from parsed commands |
| `blender_addon_server.py` | Runs inside Blender; lets an outside process request builds |
| `blender_mcp_server.py` | MCP server: lets a real AI (not model.py) call Blender by name |
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
