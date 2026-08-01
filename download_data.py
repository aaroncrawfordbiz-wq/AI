"""
The model's window to the internet: this script downloads training text from
the web, so the AI learns from real-world data instead of a hardcoded string.

Built-in datasets:
  shakespeare  ~1 MB of Shakespeare's plays — learns English style fast
  code         ~500 KB of real Python source from the official Python standard
               library — teaches the model to write Python-shaped code
  url <link>   ANY text page you point it at (e.g. a free Project Gutenberg
               book) becomes a dataset

Usage:
  python download_data.py shakespeare
  python download_data.py code
  python download_data.py url https://www.gutenberg.org/files/11/11-0.txt alice
"""

import os
import sys
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

SHAKESPEARE_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                   "master/data/tinyshakespeare/input.txt")

# Real, battle-tested Python from the standard library (pinned to a fixed
# version so this list stays stable).
CPYTHON = "https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/"
CODE_FILES = [
    "argparse.py", "dataclasses.py", "pathlib.py", "textwrap.py",
    "difflib.py", "statistics.py", "queue.py", "heapq.py", "bisect.py",
    "functools.py", "string.py", "uuid.py", "json/__init__.py",
    "json/decoder.py", "json/encoder.py",
]


def fetch(url):
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", errors="ignore")


def get_dataset(name, url=None):
    """Download (if not already cached in data/) and return the path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name}.txt")
    if os.path.exists(path):
        print(f"already downloaded: {path}")
        return path

    if name == "shakespeare":
        text = fetch(SHAKESPEARE_URL)
    elif name == "code":
        text = "\n\n".join(fetch(CPYTHON + f) for f in CODE_FILES)
    elif url:
        text = fetch(url)
    else:
        raise SystemExit(f"unknown dataset '{name}' — use shakespeare, code, "
                         f"or: python download_data.py url <link> <name>")

    # encoding='utf-8' matters: without it, Windows writes in its legacy
    # locale encoding and crashes on characters like 'Ł' in the code dataset
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"saved {len(text):,} characters to {path}")
    return path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "url":
        link = sys.argv[2]
        name = sys.argv[3] if len(sys.argv) > 3 else "custom"
        get_dataset(name, url=link)
    else:
        get_dataset(sys.argv[1])
