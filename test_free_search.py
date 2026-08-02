"""
Proves free_search.py's parsing and agreement logic is real, using a saved
sample of DuckDuckGo's actual HTML result structure (no live internet
needed for this test -- see free_search.py's docstring for why the live
fetch itself couldn't be proven in the environment that built this).

  python test_free_search.py
"""

from free_search import parse_results, find_agreement, answer

# A realistic slice of DuckDuckGo's HTML result page structure -- three
# results that roughly agree (real Japan population figures, worded
# differently by different sites) and two that don't (off-topic, and a
# stale unrelated number), to prove the agreement logic actually
# discriminates rather than just picking the first result.
SAMPLE_HTML = """
<div class="result">
  <a class="result__a" href="https://worldometer.example">Japan Population 2026 - Worldometer</a>
  <a class="result__snippet">The current population of Japan is <b>123,294,513</b> as of Tuesday, based on the latest UN data.</a>
</div>
<div class="result">
  <a class="result__a" href="https://wiki.example">Demographics of Japan - Encyclopedia</a>
  <a class="result__snippet">As of 2026, Japan's population is approximately <b>123.3 million</b> people, based on the latest United Nations estimates.</a>
</div>
<div class="result">
  <a class="result__a" href="https://statsjapan.example">Japan Statistics Bureau</a>
  <a class="result__snippet">Japan's population was estimated at <b>123,294,000</b> people according to United Nations data for the current year.</a>
</div>
<div class="result">
  <a class="result__a" href="https://oldnews.example">Japan Population 2010 Census (archived)</a>
  <a class="result__snippet">The 2010 census recorded Japan's population at 128,057,352, a figure now over a decade out of date.</a>
</div>
<div class="result">
  <a class="result__a" href="https://unrelated.example">Japan Travel Guide - Best Cities to Visit</a>
  <a class="result__snippet">Tokyo, Osaka, and Kyoto are among the most popular destinations for tourists visiting Japan.</a>
</div>
"""


def test_parsing():
    results = parse_results(SAMPLE_HTML)
    assert len(results) == 5, f"expected 5 results, got {len(results)}"
    assert "Worldometer" in results[0][0]
    assert "123,294,513" in results[0][1]
    print(f"PASS parsing — extracted {len(results)} (title, snippet) pairs correctly")


def test_agreement_finds_the_real_consensus():
    results = parse_results(SAMPLE_HTML)
    snippet, supporters = find_agreement(results)
    assert snippet is not None, "should have found agreement among the 3 population results"
    assert len(supporters) >= 2, f"expected >=2 supporters, got {supporters}"
    # the agreed answer must be about population, not the unrelated travel result
    assert "population" in snippet.lower() or "123" in snippet
    print(f"PASS agreement detection — {len(supporters)} sources agreed: {supporters}")


def test_no_agreement_is_honest():
    # only the two DISagreeing results -- old census + travel guide.
    # No real agreement exists, so it must say so, not force a fake answer.
    lone_results = parse_results(SAMPLE_HTML)[3:]
    snippet, supporters = find_agreement(lone_results)
    assert snippet is None, "should NOT fabricate agreement between unrelated snippets"
    print("PASS honesty check — correctly reports no confident answer when none exists")


def test_full_pipeline_via_monkeypatch():
    import free_search
    original = free_search.fetch_results
    free_search.fetch_results = lambda q, n=5: parse_results(SAMPLE_HTML, n)
    try:
        result = answer("population of japan")
        print(f"answer() output:\n{result}\n")
        assert "123" in result and "matched across" in result
        print("PASS full answer() pipeline produces a real, sourced answer")
    finally:
        free_search.fetch_results = original


if __name__ == "__main__":
    test_parsing()
    test_agreement_finds_the_real_consensus()
    test_no_agreement_is_honest()
    test_full_pipeline_via_monkeypatch()
    print("\nall tests passed")
