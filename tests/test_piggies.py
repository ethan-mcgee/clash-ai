"""Every part of Piggies EQ runs in the engine: all eight cards, both evolutions, and the
Archer Queen's ability, with nothing refused or left unmodelled."""

from collections import Counter

import numpy as np
import pytest

from clashai.decks import PIGGIES_EQ
from clashai.sim.env import make_vector
from clashai.sim.policies import RandomLegal

pytestmark = pytest.mark.engine


def test_piggies_eq_plays_every_card_evolution_and_ability():
    n = 32
    with make_vector(n) as v:
        s, h = v.schema, v.schema.header
        slots = PIGGIES_EQ.slots(s.slot_evolution)
        v.reset(list(range(n)), list(PIGGIES_EQ.cards), list(PIGGIES_EQ.cards), slots, slots)
        name = {v.card_id(c): c for c in PIGGIES_EQ.cards}
        # Patient enough to save up for Royal Hogs (5 elixir) and charge its evolution.
        policy = RandomLegal(0.05)
        policy.bind(v)
        rng = np.random.default_rng(0)
        plays, evolved, abilities = Counter(), Counter(), 0
        finished = np.zeros(n, bool)
        illegal = not_modelled = 0
        while not finished.all():
            actions = policy.act(v.observations.reshape(2 * n, -1), v.masks.reshape(2 * n, -1),
                                 rng).reshape(n, 2)
            actions[finished] = s.wait
            for i, seat in zip(*np.nonzero(actions), strict=True):
                a = int(actions[i, seat])
                if s.is_ability(a):
                    abilities += 1
                    continue
                slot = s.decode(a)[0]
                o = v.observations[i, seat]
                card = name[int(o[h["hand_card"][slot]])]
                plays[card] += 1
                if int(o[h["hand_form"][slot]]) == s.form_evolved:
                    evolved[card] += 1
            v.actions[:] = actions
            v.step()
            live = ~finished
            illegal += int(v.infos[live][:, [s.info["illegal_0"], s.info["illegal_1"]]].sum())
            not_modelled += int(v.infos[live, s.info["not_modelled"]].sum())
            finished |= v.dones.astype(bool)
        assert v.fault_kind_counts() == {"refusal": 0, "bug": 0, "env_bug": 0}
        assert illegal == 0 and not_modelled == 0
        assert set(plays) == set(PIGGIES_EQ.cards), plays
        assert evolved["RoyalHogs"] > 0 and evolved["Cannon"] > 0, evolved
        assert set(evolved) <= {"RoyalHogs", "Cannon"}, evolved
        assert abilities > 0
