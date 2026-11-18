"""Session mining, before the encoder gets a veto."""

from search_dojo.corpus import load_logs
from search_dojo.logs import mean_click_rank, reformulations, zero_result_queries


def test_reformulations_and_zero_results() -> None:
    events = load_logs()
    pairs = {(item.before, item.after, item.click) for item in reformulations(events)}
    assert ("white pig soup", "tonkotsu pork bone", "tonkotsu") in pairs
    assert ("milky soup", "bleach spill", "detergent") in pairs
    assert "quantum ramen" in zero_result_queries(events)
    assert mean_click_rank(events) > 1.0
