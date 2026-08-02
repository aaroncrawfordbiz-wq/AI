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
  library             ~9 MB, 18 full public-domain novels — a real English
                      text library, not just one small play.
  everything          library + every programming language combined — one
                      big brain that reads prose AND writes code.
  url <link>          ANY text page you point it at (e.g. a free Project
                      Gutenberg book) becomes a dataset

Usage:
  python download_data.py shakespeare
  python download_data.py code
  python download_data.py languages
  python download_data.py lang rust
  python download_data.py polyglot python rust go
  python download_data.py polyglot
  python download_data.py library
  python download_data.py everything
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

# 18 full public-domain novels (all pre-1929, U.S. public domain), mirrored
# on GitHub by the GITenberg project as plain text — verified reachable.
BOOK_URLS = {
    "warandpeace": "https://raw.githubusercontent.com/GITenberg/War-and-Peace_2600/master/2600.txt",
    "alice": "https://raw.githubusercontent.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/master/11.txt",
    "sherlock": "https://raw.githubusercontent.com/GITenberg/The-Adventures-of-Sherlock-Holmes_1661/master/1661.txt",
    "frankenstein": "https://raw.githubusercontent.com/GITenberg/Frankenstein_84/master/84.txt",
    "prideandprejudice": "https://raw.githubusercontent.com/GITenberg/Pride-and-Prejudice_1342/master/1342.txt",
    "greatexpectations": "https://raw.githubusercontent.com/GITenberg/Great-Expectations_1400/master/1400.txt",
    "dracula": "https://raw.githubusercontent.com/GITenberg/Dracula_345/master/345.txt",
    "janeeyre": "https://raw.githubusercontent.com/GITenberg/Jane-Eyre_1260/master/1260.txt",
    "tomsawyer": "https://raw.githubusercontent.com/GITenberg/The-Adventures-of-Tom-Sawyer_74/master/74.txt",
    "countofmontecristo": "https://raw.githubusercontent.com/GITenberg/The-Count-of-Monte-Cristo_1184/master/1184.txt",
    "metamorphosis": "https://raw.githubusercontent.com/GITenberg/Metamorphosis_5200/master/5200.txt",
    "warofworlds": "https://raw.githubusercontent.com/GITenberg/The-War-of-the-Worlds_36/master/36.txt",
    "picturedorian": "https://raw.githubusercontent.com/GITenberg/The-Picture-of-Dorian-Gray_174/master/174.txt",
    "iliad": "https://raw.githubusercontent.com/GITenberg/The-Iliad_6130/master/6130.txt",
    "crimeandpunishment": "https://raw.githubusercontent.com/GITenberg/Crime-and-Punishment_2554/master/2554.txt",
    "middlemarch": "https://raw.githubusercontent.com/GITenberg/Middlemarch_145/master/145.txt",
    "peterpan": "https://raw.githubusercontent.com/GITenberg/Peter-Pan_16/master/16.txt",
    "anneofgreengables": "https://raw.githubusercontent.com/GITenberg/Anne-of-Green-Gables_45/master/45.txt",
    "secretgarden": "https://raw.githubusercontent.com/GITenberg/The-Secret-Garden_113/master/113.txt",
}

# Real, full public-domain books on business strategy, pricing, and
# competitive markets — teaches business-style vocabulary and tone. (Style
# only, same as everything else here: it will learn to SOUND like business
# writing, not to actually analyze a market or price a product.)
BUSINESS_URLS = {
    "mylifeandwork": "https://raw.githubusercontent.com/GITenberg/My-Life-and-Work_7213/master/7213.txt",
    "artofwar": "https://raw.githubusercontent.com/GITenberg/The-Art-of-War_132/master/132.txt",
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
    elif name == "library":
        return get_library()
    elif name == "business":
        return get_business()
    elif name == "everything":
        return get_everything()
    elif name == "polyglot":
        return get_polyglot()
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


def get_library():
    """Download 18 full novels and concatenate them into one text library —
    too much text for a model to memorize, so it's forced to generalize."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "library.txt")
    if os.path.exists(path):
        print(f"already downloaded: {path}")
        return path
    parts = [fetch(u) for u in BOOK_URLS.values()]
    text = "\n\n".join(parts)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"saved {len(text):,} characters ({len(BOOK_URLS)} books) to {path}")
    return path


def get_everything():
    """The library plus every programming language — one big brain that
    reads prose and writes code."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "everything.txt")
    if os.path.exists(path):
        print(f"already downloaded: {path}")
        return path
    lib_path = get_library()
    poly_path = get_polyglot()
    text = open(lib_path, encoding="utf-8").read() + open(poly_path, encoding="utf-8").read()
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"saved {len(text):,} characters to {path}")
    return path


def get_business():
    """Real public-domain business/strategy books — teaches business-style
    writing tone and vocabulary (NOT real market data or analysis)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "business.txt")
    if os.path.exists(path):
        print(f"already downloaded: {path}")
        return path
    parts = [fetch(u) for u in BUSINESS_URLS.values()]
    text = "\n\n".join(parts)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"saved {len(text):,} characters ({len(BUSINESS_URLS)} books) to {path}")
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
    elif cmd == "library":
        get_library()
    elif cmd == "business":
        get_business()
    elif cmd == "everything":
        get_everything()
    else:
        get_dataset(cmd)
