import pytest

from clashai.data.cards import TOWERS, CardNames, split_key
from clashai.data.download import dataset_dir
from clashai.sim.env import make_vector


def test_suffixes_name_the_slot():
    assert split_key("royal-hogs-ev1") == ("royal-hogs", "evolution")
    assert split_key("knight-hero") == ("knight", "hero")
    assert split_key("the-log") == ("the-log", "")


@pytest.fixture(scope="module")
def names():
    with make_vector(1, 1) as v:
        return CardNames(v.vocabulary[i] for i in v.card_types())


@pytest.mark.engine
def test_piggies_eq_keys_map_to_its_crforge_cards(names):
    keys = ["royal-hogs-ev1", "archer-queen", "cannon-ev1", "skeletons", "ice-spirit", "the-log",
            "earthquake", "royal-delivery"]
    cards, forms = names.deck(keys)
    assert cards == ["RoyalHogs", "ArcherQueen", "Cannon", "Skeletons", "IceSpirits", "Log",
                     "Earthquake", "RoyalDelivery"]
    assert forms == ["evolution", "", "evolution", "", "", "", "", ""]


@pytest.mark.engine
def test_every_card_and_tower_in_the_replays_exists_in_crforge(names):
    if not (dataset_dir() / "replays").is_dir():
        pytest.skip("replay dataset not downloaded (python -m clashai.data.download)")
    from clashai.data.replays import read_replays

    keys, towers = set(), set()
    for i, r in enumerate(read_replays(dataset_dir())):
        for p in (r.team, r.opponent):
            keys.update(p.deck)
            towers.add(p.tower)
        if i == 20_000:
            break
    missing = [k for k in sorted(keys) if _missing(names, k)]
    assert not missing
    assert towers <= set(TOWERS)


def _missing(names, key):
    try:
        names.name(key)
        return False
    except KeyError:
        return True
