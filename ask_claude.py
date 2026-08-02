"""
A REAL AI, on demand — this file is what "my AI will ask you" actually means
in working code. It has nothing to do with model.py/model_torch.py: instead
of the tiny trained brain guessing, this makes an actual call to a genuinely
capable model (Claude) with REAL live web search, and returns its real
answer. This is the honest fix for "current market info" — the small model
can never have it (a saved checkpoint cannot update itself); a live call to
a real AI, made at the moment you ask, can.

Setup (on your own machine — this needs real internet access, which the
environment that built this file did not have; see the note below):
  pip install anthropic
  set the ANTHROPIC_API_KEY environment variable (get one at
  https://console.anthropic.com/settings/keys), OR run `ant auth login` if
  you have the Anthropic CLI — either lets this file authenticate.

Usage:
  python ask_claude.py "what is bitcoin trading at right now"
  python ask_claude.py "what's driving semiconductor stock prices this week"

IMPORTANT — this was built and unit-tested in a locked-down sandbox that
blocks outbound network calls to api.anthropic.com, so the live call path
could not be exercised end-to-end here. The request is built exactly per
Anthropic's documented Python SDK and web-search tool shape. On a normal
machine with a real API key and normal internet access, this should work
as written — if it doesn't, the error message will show the real API
error to debug from.
"""

import os
import sys

MODEL = "claude-opus-5"


def ask(question, model=MODEL, max_tokens=2048):
    """Send `question` to a real Claude model with live web search enabled.
    Returns the answer text, or a clear error string — never fake data."""
    try:
        import anthropic
    except ImportError:
        return "error: the anthropic package isn't installed — run: pip install anthropic"

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return ("error: no ANTHROPIC_API_KEY set — get a free key at "
               "https://console.anthropic.com/settings/keys and set it as "
               "an environment variable (or run `ant auth login` if you "
               "have the Anthropic CLI)")

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            tools=[{"type": "web_search_20260209", "name": "web_search"}],
            messages=[{"role": "user", "content": question}],
        )
    except anthropic.AuthenticationError:
        return "error: the ANTHROPIC_API_KEY set isn't valid — check the key"
    except anthropic.APIConnectionError as e:
        return f"error: couldn't reach Claude's API ({e}) — check your internet connection"
    except anthropic.APIStatusError as e:
        return f"error: API returned {e.status_code}: {e.message}"

    if response.stop_reason == "refusal":
        return "Claude declined to answer this request."

    # A response can interleave text with server_tool_use / web_search_tool_result
    # blocks (the actual search happening) -- we only want the final text Claude wrote.
    text = "".join(b.text for b in response.content if b.type == "text")
    return text or "(no text in response — check response.content directly)"


def looks_like_needs_real_ai(text):
    """Heuristic: does this look like a question that genuinely needs current,
    real-world information (not just a text-continuation prompt)? Used to
    decide whether to hand off to ask() instead of the trained model."""
    t = text.strip().lower()
    triggers = (
        "current", "right now", "today", "this week", "latest", "recent",
        "market", "price of", "stock", "trading at", "news", "what's happening",
    )
    return any(trigger in t for trigger in triggers)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("question> ")
    print(ask(question))
