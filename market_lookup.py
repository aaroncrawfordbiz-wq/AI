"""
A REAL market-data tool — actual live numbers fetched over the internet at
the moment you ask, never the trained model guessing.

Why this file exists: no trained model, however new, can have "current"
market info, because training freezes its knowledge the moment training
ends. There is no way to make a checkpoint "update every time you talk to
it" — a saved file of numbers does not change itself. The only honest way
to get a genuinely current answer is what this file does: skip the model
entirely and call a real live data source right now, the same trick as
calculator.py for math. This IS the update-every-time you asked for — it
just lives in a network call, not in retraining.

What's wired up (all free, no account needed unless noted):
  crypto price      CoinGecko            "price of bitcoin"
  currency exchange  Frankfurter.app      "usd to eur"
  stock quote        Alpha Vantage        "stock price AAPL"
                      (needs a free API key — see get_stock_quote())

IMPORTANT — this file could not be tested against the live internet from
the environment that built it (a locked-down sandbox that blocks most
outbound domains). It uses plain, standard, well-documented public APIs
with ordinary urllib requests, and its error handling was verified against
a real unreachable-network failure. It should work directly on a normal
PC with normal internet access — if a specific call fails for you, the
error message will show the real HTTP/network error to debug from.
"""

import json
import os
import urllib.error
import urllib.request

TIMEOUT = 10


def _get_json(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "market-lookup/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def get_crypto_price(coin, vs="usd"):
    """Real, live crypto price via CoinGecko's free public API."""
    coin = coin.lower().strip()
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies={vs}"
    try:
        data = _get_json(url)
        if coin not in data:
            return {"error": f"unknown coin id '{coin}' — try the full name, "
                             f"e.g. 'bitcoin' not 'btc'"}
        return {"coin": coin, "currency": vs, "price": data[coin][vs], "source": "CoinGecko"}
    except urllib.error.URLError as e:
        return {"error": f"couldn't reach CoinGecko ({e}) — check your internet connection"}


def get_exchange_rate(from_currency, to_currency):
    """Real, live currency exchange rate via Frankfurter (European Central
    Bank data, free, no key)."""
    f, t = from_currency.upper().strip(), to_currency.upper().strip()
    url = f"https://api.frankfurter.app/latest?from={f}&to={t}"
    try:
        data = _get_json(url)
        if t not in data.get("rates", {}):
            return {"error": f"unknown currency pair '{f}' to '{t}'"}
        return {"from": f, "to": t, "rate": data["rates"][t],
                "date": data["date"], "source": "Frankfurter / European Central Bank"}
    except urllib.error.URLError as e:
        return {"error": f"couldn't reach the exchange-rate service ({e}) — "
                         f"check your internet connection"}


def get_stock_quote(symbol):
    """Real, live stock quote via Alpha Vantage. Needs a free API key — get
    one at https://www.alphavantage.co/support/#api-key and set it as the
    ALPHAVANTAGE_API_KEY environment variable. Without one, this honestly
    says so instead of returning fake data."""
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        return {"error": "no stock API key set — get a free one at "
                         "https://www.alphavantage.co/support/#api-key and "
                         "set ALPHAVANTAGE_API_KEY (environment variable)"}
    symbol = symbol.upper().strip()
    url = (f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE"
          f"&symbol={symbol}&apikey={key}")
    try:
        data = _get_json(url).get("Global Quote", {})
        if not data:
            return {"error": f"no data for symbol '{symbol}' — check it's a "
                             f"valid ticker, or the API key's rate limit was hit"}
        return {"symbol": symbol, "price": data.get("05. price"),
                "change_percent": data.get("10. change percent"),
                "source": "Alpha Vantage"}
    except urllib.error.URLError as e:
        return {"error": f"couldn't reach Alpha Vantage ({e})"}


def looks_like_market_query(text):
    """Heuristic: does this look like a request for real market data rather
    than a prompt for the language model? Returns (kind, args) or None."""
    t = text.strip().lower().rstrip("?")

    for coin in ("bitcoin", "ethereum", "dogecoin", "litecoin", "solana", "cardano"):
        if coin in t or (coin[:3] in t and "price" in t):
            return ("crypto", coin)

    import re
    m = re.search(r"\b([a-z]{3})\s*(?:to|->|in)\s*([a-z]{3})\b", t)
    if m and ("exchange" in t or "convert" in t or "to" in t):
        return ("exchange", (m.group(1), m.group(2)))

    m = re.search(r"stock\s+(?:price\s+)?(?:of\s+|for\s+)?([a-z]{1,5})\b", t)
    if m:
        return ("stock", m.group(1))

    return None


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or input("query> ")
    kind = looks_like_market_query(query)
    if kind is None:
        print("not recognized as a market query")
    elif kind[0] == "crypto":
        print(get_crypto_price(kind[1]))
    elif kind[0] == "exchange":
        print(get_exchange_rate(*kind[1]))
    elif kind[0] == "stock":
        print(get_stock_quote(kind[1]))
