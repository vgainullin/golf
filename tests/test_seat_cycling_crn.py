"""Common random numbers in seat cycling: with a deal seed, every hole is dealt the same cards
whatever the players did before it."""
import torch

import scripts.seat_cycling as sc


def _deals(monkeypatch, seating, deal_seed, holes=3):
    seen = []
    real = sc.reset_games

    def rec(*a, **kw):
        state = real(*a, **kw)
        seen.append(state.deck.clone())
        return state

    monkeypatch.setattr(sc, "reset_games", rec)
    torch.manual_seed(0)
    sc.run_seating(seating, 8, holes, torch.device("cpu"), deal_seed=deal_seed)
    return seen


def test_deal_seed_pins_every_hole(monkeypatch):
    # Random players consume RNG during play; the improved heuristic does not.
    a = _deals(monkeypatch, ("R", "R", "R", "R"), deal_seed=7)
    b = _deals(monkeypatch, ("I", "I", "I", "I"), deal_seed=7)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_without_deal_seed_later_holes_drift(monkeypatch):
    a = _deals(monkeypatch, ("R", "R", "R", "R"), deal_seed=None)
    b = _deals(monkeypatch, ("I", "I", "I", "I"), deal_seed=None)
    assert torch.equal(a[0], b[0])  # same global seed, same first deal
    assert not torch.equal(a[1], b[1])  # play consumed RNG differently


def test_seatings_get_different_deals(monkeypatch):
    a = _deals(monkeypatch, ("I", "I", "I", "I"), deal_seed=7 * 1000 + 0)
    b = _deals(monkeypatch, ("I", "I", "I", "I"), deal_seed=7 * 1000 + 1)
    assert not torch.equal(a[0], b[0])
