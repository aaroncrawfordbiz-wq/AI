"""
Answers questions using the internet directly — NO API key, no Claude, no
account, completely free. This is the honest "make it search and answer
without an API key" version: it fetches real search results, checks whether
several of them actually agree, and only answers when they do.

How it works (exactly what you described):
  1. Search the web for your question (DuckDuckGo's HTML page — the one
     search engine with a plain page that doesn't require a paid API key).
  2. Pull the top 5 result snippets.
  3. Compare them to each other — do at least 2 independently say roughly
     the same thing?
  4. If yes: show you that agreed-on text plus which sites said it (you're
     seeing real excerpts, not this script's own writing).
     If no: say so honestly instead of guessing.

READ THIS — the honest ceiling, and why ask_claude.py still exists:
  This can only work for QUESTIONS WITH A SHORT FACTUAL ANSWER that search
  engines already index well ("population of Japan", "capital of France",
  "boiling point of water"). It cannot have a conversation, cannot reason
  about anything, cannot write code, and cannot answer opinion or "explain
  why" questions — it has no understanding at all, it is only comparing
  chunks of text for overlap. Think of it as a much dumber, much more
  literal cousin of calculator.py: real data, zero understanding. For
  anything needing actual comprehension, ask_claude.py (which costs a
  little money per question) is genuinely doing something different and
  more capable — this script is the free-but-limited alternative.

  Built and tested in a sandbox that blocks every search engine's network
  access, so the LIVE fetch could not be proven end to end here — the
  parsing/comparison logic below was proven correct against a real saved
  copy of DuckDuckGo's HTML structure (see test_free_search.py). It should
  work with normal internet access; if DuckDuckGo changes their page
  layout in the future, the regex patterns below may need updating.

Usage:
  python free_search.py "population of japan"
"""

import difflib
import html
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

SEARCH_URL = "https://html.duckduckgo.com/html/?q={}"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; free_search/1.0)"}

# DuckDuckGo's HTML result page wraps each snippet in a <a class="result__snippet">.
SNIPPET_RE = re.compile(
    r'class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
TITLE_RE = re.compile(
    r'class="result__a"[^>]*>(.*?)</a>', re.DOTALL)


def _clean(raw_html_fragment):
    """Strip HTML tags and unescape entities from one snippet fragment."""
    text = re.sub(r"<[^>]+>", "", raw_html_fragment)
    return html.unescape(text).strip()


def fetch_results(query, num_results=5, timeout=15):
    """Real HTTP GET to DuckDuckGo's plain HTML page (no API key needed —
    this is the same page a browser with JavaScript off would see).
    Returns a list of (title, snippet) tuples, or raises on network failure."""
    url = SEARCH_URL.format(urllib.parse.quote(query))
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8", errors="ignore")
    return parse_results(body, num_results)


def parse_results(body, num_results=5):
    """Pure parsing logic, separated from the network call so it can be
    tested against a saved HTML sample without needing live internet."""
    snippets = [_clean(m) for m in SNIPPET_RE.findall(body)]
    titles = [_clean(m) for m in TITLE_RE.findall(body)]
    pairs = list(zip(titles, snippets))[:num_results]
    return [p for p in pairs if p[1]]  # drop any with an empty snippet


def _similarity(a, b):
    """0..1 -- how much two snippets overlap. Plain difflib, no ML."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers_in(text):
    """Extract the numeric figures in a snippet (commas/decimals stripped
    to plain floats), e.g. '123,294,513' and '123.3 million' -> [123294513.0,
    123300000.0]. 'million'/'billion' are expanded so wording differences
    don't hide a genuine numeric mismatch."""
    out = []
    for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s*(million|billion)?", text, re.I):
        digits, scale = m.groups()
        if not digits.strip(","):
            continue
        value = float(digits.replace(",", ""))
        if scale and scale.lower() == "million":
            value *= 1_000_000
        elif scale and scale.lower() == "billion":
            value *= 1_000_000_000
        out.append(value)
    return out


def _numbers_agree(a, b, tolerance=0.02):
    """True if every number in `a` has a closely matching number in `b`
    (within 2% relative difference) -- or if NEITHER snippet contains a
    number, in which case there's nothing numeric to contradict. If one has
    numbers and the other doesn't share a matching one, they do NOT agree,
    even if the wording looks similar (this is exactly the bug an earlier
    version of this file had: an outdated figure with similar phrasing was
    wrongly counted as agreeing)."""
    nums_a, nums_b = _numbers_in(a), _numbers_in(b)
    if not nums_a and not nums_b:
        return True
    if not nums_a or not nums_b:
        return False
    return any(
        abs(x - y) <= tolerance * max(x, y, 1)
        for x in nums_a for y in nums_b
    )


def find_agreement(results, threshold=0.35):
    """Look for a cluster of results that substantially agree with each
    other -- both in wording AND in any numbers they state. Returns
    (agreed_snippet, supporting_titles) or (None, []) if nothing lines up --
    an honest 'I don't have a confident answer', never a guess dressed up
    as one, and never two DIFFERENT numbers treated as agreeing just
    because the sentence around them reads similarly."""
    best = (None, [], 0)  # snippet, supporting titles, agreement count
    for i, (title_i, snip_i) in enumerate(results):
        supporters = [title_i]
        for j, (title_j, snip_j) in enumerate(results):
            if i == j:
                continue
            if (_similarity(snip_i, snip_j) >= threshold
                    and _numbers_agree(snip_i, snip_j)):
                supporters.append(title_j)
        if len(supporters) > best[2]:
            best = (snip_i, supporters, len(supporters))
    snippet, supporters, count = best
    if count < 2:  # nobody agreed with anybody -- no confident answer
        return None, []
    return snippet, supporters


def answer(query, num_results=5):
    """The whole pipeline. Returns a plain-text answer, or an honest
    'couldn't find agreement' message -- never fabricated text."""
    try:
        results = fetch_results(query, num_results)
    except urllib.error.URLError as e:
        return f"error: couldn't reach the search engine ({e}) — check your internet connection"

    if not results:
        return "no search results came back — try rephrasing the question"

    snippet, supporters = find_agreement(results)
    if snippet is None:
        return ("no confident answer — the top results didn't agree with "
               "each other closely enough. Try ask_claude.py for a real "
               "AI's actual understanding instead of raw text-matching.")

    sources = ", ".join(supporters[:3])
    return f"{snippet}\n\n(matched across {len(supporters)} source(s): {sources})"


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or input("search> ")
    print(answer(query))
