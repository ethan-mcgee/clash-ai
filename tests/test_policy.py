"""How a network policy turns logits into the distributions it samples from."""

import torch

from clashai.model.policy import tile_probs, type_probs


def _logits(rows=6, cols=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, cols, generator=g) * 2
    mask = torch.rand(rows, cols, generator=g) < 0.7
    mask[:, 0] = True  # the wait is always open
    mask[0, 1:] = False  # one row where only the wait is open
    return logits, mask


def test_at_one_both_are_the_plain_masked_softmax():
    logits, mask = _logits()
    plain = torch.softmax(logits.masked_fill(~mask, -1e9), -1)
    torch.testing.assert_close(type_probs(logits, mask), plain)
    torch.testing.assert_close(tile_probs(logits, mask), plain)


def test_card_temperature_keeps_the_chance_of_playing_and_sharpens_the_choice():
    logits, mask = _logits()
    plain = type_probs(logits, mask)
    sharp = type_probs(logits, mask, card_temperature=0.5)
    torch.testing.assert_close(sharp[:, 0], plain[:, 0])
    torch.testing.assert_close(sharp.sum(1), torch.ones(len(sharp)))
    assert (sharp[~mask] == 0).all()
    assert sharp[0, 0] == 1  # only the wait open: nothing to choose
    acts = mask[:, 1:].sum(1) >= 2
    best = (logits[:, 1:].masked_fill(~mask[:, 1:], -1e9)).argmax(1)
    rows = torch.arange(len(logits))
    # The likeliest play gets a larger share of the playing probability.
    share = lambda p: p[rows, best + 1] / (1 - p[:, 0])
    assert (share(sharp)[acts] > share(plain)[acts]).all()


def test_top_p_keeps_the_fewest_likeliest_tiles_holding_the_share():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05], [0.95, 0.03, 0.01, 0.01]]))
    mask = torch.ones_like(logits, dtype=torch.bool)
    p = tile_probs(logits, mask, top_p=0.8)
    # Row 0: 0.5 + 0.3 reaches 0.8, so two tiles; row 1: the first alone holds it.
    torch.testing.assert_close(p[0], torch.tensor([0.625, 0.375, 0.0, 0.0]))
    torch.testing.assert_close(p[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    closed = mask.clone()
    closed[0, 0] = False  # the likeliest tile closed: the rest renormalise among the open
    q = tile_probs(logits, closed, top_p=0.8)
    assert q[0, 0] == 0 and abs(q[0].sum() - 1) < 1e-6


def test_tile_temperature_below_one_sharpens():
    logits, mask = _logits(cols=40)
    plain, sharp = tile_probs(logits, mask), tile_probs(logits, mask, temperature=0.5)
    assert (sharp.max(1).values >= plain.max(1).values).all()
    assert (sharp[~mask] == 0).all()


def test_at_temperature_zero_card_and_tile_are_the_most_likely_and_the_wait_keeps_its_chance():
    logits, mask = _logits()
    plain = type_probs(logits, mask)
    best = type_probs(logits, mask, card_temperature=0.0)
    torch.testing.assert_close(best[:, 0], plain[:, 0])
    torch.testing.assert_close(best.sum(1), torch.ones(len(best)))
    top = logits[:, 1:].masked_fill(~mask[:, 1:], float("-inf")).argmax(1)
    acts = mask[:, 1:].any(1)
    rows = torch.arange(len(logits))[acts]
    torch.testing.assert_close(best[rows, top[acts] + 1], 1 - plain[rows, 0])
    assert best[0, 0] == 1  # only the wait open
    tiles = tile_probs(logits, mask, temperature=0.0)
    assert (tiles.sum(1) == 1).all() and (tiles.max(1).values == 1).all()
    assert (tiles.argmax(1) == logits.masked_fill(~mask, float("-inf")).argmax(1)).all()
