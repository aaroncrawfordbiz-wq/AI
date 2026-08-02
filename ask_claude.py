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
  python ask_claude.py "what is bitcoin trading at right now"     # one question
  python ask_claude.py --chat                                     # real back-and-forth chat

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


def _client_or_error():
    """Shared setup: returns (client, None) or (None, error_string)."""
    try:
        import anthropic
    except ImportError:
        return None, "error: the anthropic package isn't installed — run: pip install anthropic"

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return None, ("error: no ANTHROPIC_API_KEY set — get a free key at "
                      "https://console.anthropic.com/settings/keys and set it as "
                      "an environment variable (or run `ant auth login` if you "
                      "have the Anthropic CLI)")
    return anthropic.Anthropic(), None


def _extract_text(response):
    # A response can interleave text with server_tool_use / web_search_tool_result
    # blocks (the actual search happening) -- we only want the final text Claude wrote.
    return "".join(b.text for b in response.content if b.type == "text")


def ask(question, model=MODEL, max_tokens=2048):
    """Send ONE question to a real Claude model with live web search enabled,
    no memory of anything before it. Returns the answer text, or a clear
    error string — never fake data."""
    import anthropic
    client, err = _client_or_error()
    if err:
        return err

    try:
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
    return _extract_text(response) or "(no text in response — check response.content directly)"


def chat(model=MODEL, max_tokens=2048):
    """A REAL back-and-forth conversation with Claude — this is what actually
    'talking like me' means: not a bigger/longer-trained checkpoint (that
    structurally cannot become a conversational assistant — see this file's
    docstring), but a live connection to a real, already-trained assistant.
    Keeps the whole conversation history and resends it each turn, same as
    any real chat app does — the API itself has no memory between calls."""
    import anthropic
    client, err = _client_or_error()
    if err:
        print(err)
        return

    print(f"chatting with {model} — real answers, real memory of this "
          f"conversation, ctrl-c to quit\n")
    messages = []
    while True:
        try:
            user_text = input("you> ")
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if not user_text.strip():
            continue
        messages.append({"role": "user", "content": user_text})

        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                tools=[{"type": "web_search_20260209", "name": "web_search"}],
                messages=messages,
            )
        except anthropic.AuthenticationError:
            print("error: the ANTHROPIC_API_KEY set isn't valid — check the key")
            messages.pop()
            continue
        except anthropic.APIConnectionError as e:
            print(f"error: couldn't reach Claude's API ({e})")
            messages.pop()
            continue
        except anthropic.APIStatusError as e:
            print(f"error: API returned {e.status_code}: {e.message}")
            messages.pop()
            continue

        if response.stop_reason == "refusal":
            print("Claude declined to answer that.")
            messages.pop()
            continue

        text = _extract_text(response)
        print(f"\nclaude> {text}\n")
        # keep the real reply in history so it remembers this conversation
        messages.append({"role": "assistant", "content": response.content})


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
    if "--chat" in sys.argv:
        chat()
    else:
        question = " ".join(sys.argv[1:]) or input("question> ")
        print(ask(question))
