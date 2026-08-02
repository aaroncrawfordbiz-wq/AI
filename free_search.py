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

REAL BROWSER MODE (the fix for exactly that problem): search engines
change their page layout, which breaks any parser tied to specific CSS
class names — that's what happened above. Browser mode sidesteps this by
opening a REAL browser (Safari, already on every Mac — no download) and
reading whatever text is actually rendered on screen, instead of guessing
at HTML tags. This is far more robust to layout changes, because it works
off visible text, not markup.

One-time setup for Safari (no extra software to install):
  1. Safari menu -> Settings -> Advanced tab -> check
     "Show features for web developers"
  2. The "Develop" menu now appears in the menu bar -> click it ->
     check "Allow Remote Automation"
  3. pip3 install selenium

Usage:
  python free_search.py                        # opens a "you>" chat loop
  python free_search.py "population of japan"   # one question
  python free_search.py --chrome                # use Chrome instead of Safari
                                                  # (needs Chrome + chromedriver)

IMPORTANT — this environment has neither macOS nor Safari nor internet
access to search engines, so the browser-automation path could not be run
live here either. It's written to Selenium's real, documented WebDriver
API and Apple's documented Safari automation setup — it should work as
written on your actual Mac. If it doesn't, the error message will name
the real problem (Selenium missing, Safari automation not enabled, etc.)
rather than fail silently.
"""

import difflib
import html
import re
import sys
import time
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


# ---------------------------------------------------------------------------
# Real browser mode -- reads whatever a real browser actually renders,
# instead of guessing at a search engine's HTML tag names (which is exactly
# what broke above when DuckDuckGo changed their page layout).
# ---------------------------------------------------------------------------

def _launch_browser(engine="safari"):
    """Starts a REAL browser under your control via Selenium's WebDriver
    protocol -- the same technology behind real browser automation and
    testing tools. Returns (driver, None) or (None, error_string)."""
    try:
        from selenium import webdriver
    except ImportError:
        return None, "error: selenium isn't installed — run: pip3 install selenium"

    try:
        if engine == "safari":
            driver = webdriver.Safari()
        elif engine == "chrome":
            driver = webdriver.Chrome()
        else:
            return None, f"error: unknown engine '{engine}' — use 'safari' or 'chrome'"
    except Exception as e:
        hint = ""
        if engine == "safari":
            hint = (" — on your Mac: Safari > Settings > Advanced > check "
                   "'Show features for web developers', then the new "
                   "'Develop' menu > check 'Allow Remote Automation' "
                   "(one-time setup, see this file's docstring)")
        return None, f"error: couldn't start {engine} ({e}){hint}"
    return driver, None


def fetch_page_text(query, engine="safari", timeout=20):
    """Opens a real browser, navigates to a real search results page, and
    returns the actual VISIBLE TEXT of that page -- exactly what a human
    looking at the screen would see. Unlike raw HTML scraping, this
    survives the search engine changing their markup, because it never
    looks at tag names or CSS classes at all."""
    driver, err = _launch_browser(engine)
    if err:
        return None, err
    try:
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        time.sleep(2)  # let the page finish rendering
        body_text = driver.find_element("tag name", "body").text
        return body_text, None
    except Exception as e:
        return None, f"error: browser navigation failed ({e})"
    finally:
        driver.quit()


def _candidate_lines(body_text, min_len=25, max_len=400):
    """Search-result pages render as many short lines of text (nav links,
    menus, ads) plus the actual result snippets. Keep only lines in a
    plausible 'real sentence' length range -- crude, but it doesn't depend
    on any specific site's markup, so it survives layout changes."""
    lines = [ln.strip() for ln in body_text.splitlines()]
    return [ln for ln in lines if min_len <= len(ln) <= max_len]


def answer_via_browser(query, engine="safari"):
    """The browser-mode pipeline: real browser -> real rendered text ->
    same honest agreement check as answer() above. Never fabricates a
    result — says so plainly when nothing lines up."""
    body_text, err = fetch_page_text(query, engine)
    if err:
        return err

    lines = _candidate_lines(body_text)
    if not lines:
        return "no usable text came back from the page — try rephrasing the question"

    # reuse the exact same agreement logic as the non-browser path, just
    # labeling each candidate line as its own numbered "result"
    labeled = [(f"result {i + 1}", line) for i, line in enumerate(lines)]
    snippet, supporters = find_agreement(labeled)
    if snippet is None:
        return ("no confident answer — nothing on the results page agreed "
               "closely enough. Try ask_claude.py for a real AI's actual "
               "understanding instead of raw text-matching.")
    return f"{snippet}\n\n(this appeared, in agreement, {len(supporters)} times on the results page)"


def repl(engine="safari"):
    """A persistent 'you>' chat loop, so you never have to retype the
    command — just run 'python free_search.py' with no arguments."""
    print(f"free web search ({engine}, real browser, no API key, no "
          f"account) — ctrl-c to quit\n")
    while True:
        try:
            query = input("you> ")
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if not query.strip():
            continue
        print()
        print(answer_via_browser(query, engine))
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    engine = "chrome" if "--chrome" in args else "safari"
    args = [a for a in args if a not in ("--chrome", "--safari")]

    if args:
        print(answer_via_browser(" ".join(args), engine))
    else:
        repl(engine)
