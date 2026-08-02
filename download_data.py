"""
The model's window to the internet: this script downloads training text from
the web, so the AI learns from real-world data instead of a hardcoded string.

Built-in datasets:
  shakespeare        ~1 MB of Shakespeare's plays — learns English style fast
  code                real Python source from the official Python standard
                      library — teaches the model to write Python-shaped code
  lang <language>     real source code in one specific language — see the
                      full list with: python download_data.py languages
  polyglot [langs...] several languages combined into one file, so a single
                      model learns the shape of all of them (prompt with
                      "def " for Python-mode, "function " for JS-mode, etc).
                      With no language names, uses ALL of them.
  url <link>          ANY text page you point it at (e.g. a free Project
                      Gutenberg book) becomes a dataset

Usage:
  python download_data.py shakespeare
  python download_data.py code
  python download_data.py languages
  python download_data.py lang rust
  python download_data.py polyglot python rust go
  python download_data.py polyglot
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

# Real source files, one or two per language, from well-known open-source
# projects, pinned to a fixed tag/commit where the project has one so this
# list keeps working. Every URL below was checked to load before shipping.
LANG_FILES = {
    "python": [CPYTHON + f for f in CODE_FILES],
    "perl": [
        "https://raw.githubusercontent.com/Perl/perl5/v5.38.2/lib/File/Basename.pm",
    ],
    "java": [
        "https://raw.githubusercontent.com/google/gson/main/gson/src/main/java/com/google/gson/Gson.java",
        "https://raw.githubusercontent.com/google/gson/main/gson/src/main/java/com/google/gson/JsonObject.java",
        "https://raw.githubusercontent.com/square/retrofit/master/retrofit/src/main/java/retrofit2/Retrofit.java",
    ],
    "cpp": [
        "https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp",
    ],
    "csharp": [
        "https://raw.githubusercontent.com/JamesNK/Newtonsoft.Json/13.0.3/Src/Newtonsoft.Json/JsonConvert.cs",
        "https://raw.githubusercontent.com/JamesNK/Newtonsoft.Json/13.0.3/Src/Newtonsoft.Json/Linq/JObject.cs",
    ],
    "javascript": [
        "https://raw.githubusercontent.com/nodejs/node/v20.11.1/lib/fs.js",
        "https://raw.githubusercontent.com/nodejs/node/v20.11.1/lib/path.js",
    ],
    "c": [
        "https://raw.githubusercontent.com/raysan5/raylib/5.0/src/rtext.c",
    ],
    "lua": [
        "https://raw.githubusercontent.com/rxi/json.lua/master/json.lua",
        "https://raw.githubusercontent.com/kikito/inspect.lua/master/inspect.lua",
    ],
    "go": [
        "https://raw.githubusercontent.com/golang/go/go1.22.0/src/strings/strings.go",
        "https://raw.githubusercontent.com/golang/go/go1.22.0/src/sort/sort.go",
    ],
    "rust": [
        "https://raw.githubusercontent.com/BurntSushi/ripgrep/14.1.0/crates/globset/src/lib.rs",
    ],
    "ruby": [
        "https://raw.githubusercontent.com/rails/rails/v7.1.3/activesupport/lib/active_support/core_ext/string/inflections.rb",
    ],
    "php": [
        "https://raw.githubusercontent.com/laravel/framework/v11.0.0/src/Illuminate/Support/Str.php",
    ],
    "swift": [
        "https://raw.githubusercontent.com/apple/swift-algorithms/1.2.0/Sources/Algorithms/Chunked.swift",
        "https://raw.githubusercontent.com/apple/swift-algorithms/1.2.0/Sources/Algorithms/Combinations.swift",
    ],
    "shell": [
        "https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/nvm.sh",
    ],
    "bash": [
        "https://raw.githubusercontent.com/junegunn/fzf/master/install",
    ],
    "powershell": [
        "https://raw.githubusercontent.com/dahlbyk/posh-git/master/src/posh-git.psm1",
    ],
    "micropython": [
        "https://raw.githubusercontent.com/micropython/micropython-lib/master/python-stdlib/collections/collections/__init__.py",
    ],
    "matlab": [
        "https://raw.githubusercontent.com/altmany/export_fig/master/export_fig.m",
    ],
    "julia": [
        "https://raw.githubusercontent.com/JuliaLang/julia/v1.10.0/base/strings/basic.jl",
    ],
    "r": [
        "https://raw.githubusercontent.com/tidyverse/dplyr/main/R/select.R",
    ],
    "scala": [
        "https://raw.githubusercontent.com/scala/scala/v2.13.12/src/library/scala/collection/immutable/List.scala",
    ],
    "typescript": [
        "https://raw.githubusercontent.com/microsoft/TypeScript/v5.4.2/src/compiler/core.ts",
    ],
    "gdscript": [
        "https://raw.githubusercontent.com/godotengine/godot-demo-projects/master/2d/platformer/player/player.gd",
    ],
    "hlsl": [
        "https://raw.githubusercontent.com/microsoft/DirectX-Graphics-Samples/master/Samples/Desktop/D3D12HelloWorld/src/HelloTriangle/shaders.hlsl",
    ],
    "glsl": [
        "https://raw.githubusercontent.com/KhronosGroup/Vulkan-Samples/main/shaders/triangle.frag",
        "https://raw.githubusercontent.com/KhronosGroup/Vulkan-Samples/main/shaders/triangle.vert",
    ],
    "wgsl": [
        "https://raw.githubusercontent.com/gfx-rs/wgpu/trunk/examples/features/src/hello_triangle/shader.wgsl",
    ],
    "kotlin": [
        "https://raw.githubusercontent.com/JetBrains/kotlin/v1.9.22/libraries/stdlib/src/kotlin/collections/Collections.kt",
    ],
}


def fetch(url):
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", errors="ignore")


def fetch_language(lang):
    urls = LANG_FILES.get(lang)
    if not urls:
        available = ", ".join(sorted(LANG_FILES))
        raise SystemExit(f"unknown language '{lang}' — available: {available}")
    return "\n\n".join(fetch(u) for u in urls)


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
        text = fetch_language("python")
    elif name.startswith("lang_"):
        text = fetch_language(name[len("lang_"):])
    elif url:
        text = fetch(url)
    else:
        raise SystemExit(f"unknown dataset '{name}' — use shakespeare, code, "
                         f"lang <language>, polyglot, "
                         f"or: python download_data.py url <link> <name>")

    # encoding='utf-8' matters: without it, Windows writes in its legacy
    # locale encoding and crashes on characters outside its default codepage
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"saved {len(text):,} characters to {path}")
    return path


def get_polyglot(langs=None):
    """Download several languages and concatenate them into one training
    file, each labeled, so one model can learn every language's style."""
    langs = langs or sorted(LANG_FILES)
    unknown = [l for l in langs if l not in LANG_FILES]
    if unknown:
        raise SystemExit(f"unknown language(s): {', '.join(unknown)} — see: "
                         f"python download_data.py languages")

    os.makedirs(DATA_DIR, exist_ok=True)
    name = "polyglot" if langs == sorted(LANG_FILES) else "polyglot_" + "_".join(langs)
    path = os.path.join(DATA_DIR, f"{name}.txt")
    if os.path.exists(path):
        print(f"already downloaded: {path}")
        return path

    parts = []
    for lang in langs:
        body = fetch_language(lang)
        parts.append(f"\n\n# ===== {lang} =====\n\n{body}")
    text = "".join(parts)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"saved {len(text):,} characters ({len(langs)} languages) to {path}")
    return path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd = sys.argv[1]
    if cmd == "url":
        link = sys.argv[2]
        name = sys.argv[3] if len(sys.argv) > 3 else "custom"
        get_dataset(name, url=link)
    elif cmd == "lang":
        if len(sys.argv) < 3:
            raise SystemExit("usage: python download_data.py lang <language>")
        get_dataset(f"lang_{sys.argv[2]}")
    elif cmd == "languages":
        print("available languages:")
        for lang in sorted(LANG_FILES):
            print(f"  {lang}")
    elif cmd == "polyglot":
        get_polyglot(sys.argv[2:] or None)
    else:
        get_dataset(cmd)
