"""The toy tower learns a tiny paraphrase."""

from search_dojo.biencoder import TinyBiEncoder


def test_loss_falls_and_positive_wins() -> None:
    texts = [
        "cloudy pig soup",
        "opaque pork bone broth famous midnight",
        "zebra forest leaves tongue",
    ]
    encoder = TinyBiEncoder.from_texts(texts, dim=16, seed=3)
    curve = encoder.train(
        [("cloudy pig soup", texts[1], texts[2])],
        epochs=30,
        lr=0.6,
        aux_lr=0.5,
        aux_terms=4,
        df={"opaque": 1, "pork": 1, "midnight": 1, "famous": 1, "broth": 1, "bone": 1},
        anchor=0.02,
    )
    assert curve[-1] < curve[0]
    positive = encoder.score("cloudy pig soup", texts[1])
    negative = encoder.score("cloudy pig soup", texts[2])
    assert positive > negative


def test_self_cosine_is_one() -> None:
    encoder = TinyBiEncoder.from_texts(["tonkotsu broth"], dim=8, seed=1)
    assert encoder.score("tonkotsu broth", "tonkotsu broth") == 1.0
