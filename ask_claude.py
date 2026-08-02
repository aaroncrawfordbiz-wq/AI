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
  python ask_claude.py --chat --no-thinking                       # hide the [thinking] section

By default every answer is shown like this (a REAL summary of Claude's
actual reasoning, from the API's real extended-thinking feature -- never
a fabricated placeholder):
  you> hi
  claude> [thinking] The user just greeted me casually. I should greet
  them back warmly and ask how I can help. [thought]
  hi! how are you? how can I help you today?

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


def _extract_thinking(response):
    """Real thinking blocks -- Claude's actual reasoning, summarized into
    readable text (display: 'summarized'), not invented or simulated."""
    return "\n".join(b.thinking for b in response.content
                     if b.type == "thinking" and b.thinking)


def _format_reply(response, show_thinking):
    """[thinking] ... [thought] <answer> -- the format you asked for. The
    thinking text is Claude's REAL reasoning (a summary of it, since the raw
    chain of thought is never exposed by the API), not a fake placeholder."""
    out = []
    if show_thinking:
        thinking = _extract_thinking(response)
        out.append(f"[thinking] {thinking or '...'} [thought]")
    out.append(_extract_text(response) or "(no text in response)")
    return "\n".join(out)


def ask(question, model=MODEL, max_tokens=4096, show_thinking=True):
    """Send ONE question to a real Claude model with live web search enabled,
    no memory of anything before it. Returns the formatted answer (with a
    real [thinking]...[thought] section if show_thinking), or a clear error
    string — never fake data."""
    import anthropic
    client, err = _client_or_error()
    if err:
        return err

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive", "display": "summarized"},
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
    return _format_reply(response, show_thinking)


def chat(model=MODEL, max_tokens=4096, show_thinking=True):
    """A REAL back-and-forth conversation with Claude — this is what actually
    'talking like me' means: not a bigger/longer-trained checkpoint (that
    structurally cannot become a conversational assistant — see this file's
    docstring), but a live connection to a real, already-trained assistant.
    Keeps the whole conversation history and resends it each turn, same as
    any real chat app does — the API itself has no memory between calls.

    show_thinking prints Claude's REAL reasoning as [thinking] ... [thought]
    before the answer, using the API's actual extended-thinking feature
    (a genuine summary of real reasoning, never a fabricated placeholder)."""
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
                thinking={"type": "adaptive", "display": "summarized"},
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

        print(f"\nclaude> {_format_reply(response, show_thinking)}\n")
        # keep the FULL response (including thinking blocks) in history --
        # required so Claude can correctly continue reasoning next turn
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
    show_thinking = "--no-thinking" not in sys.argv
    args = [a for a in sys.argv[1:] if a not in ("--chat", "--no-thinking")]

    if "--chat" in sys.argv:
        chat(show_thinking=show_thinking)
    else:
        question = " ".join(args) or input("question> ")
        print(ask(question, show_thinking=show_thinking))
