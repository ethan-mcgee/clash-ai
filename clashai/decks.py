"""The deck we train, the reference opponent decks, and the held-out split.

Card names are crforge's (the game tables' row names): ``RoyalHogs``, ``IceSpirits``, ``Log``.
Some differ from the in-game names: ``RageBarbarian`` is the Lumberjack, ``DarkWitch`` the Night
Witch, ``Assassin`` the Bandit, ``Ghost`` the Royal Ghost, ``AxeMan`` the Executioner.

The reference decks are hand-picked archetypes for the M0 scorecard. M1's audit of the November
replays replaces them with the decks Piggies EQ actually meets, at their real rates.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Deck:
    name: str
    cards: tuple[str, ...]
    evolutions: tuple[str, ...] = ()
    archetype: str = ""
    flags: tuple[int, ...] = ()
    """crforge slot flags as recorded, one per card in deck order (a deck met in the replays,
    whose slots may also hold a hero); when given, they replace the flags ``evolutions`` makes."""

    def __post_init__(self) -> None:
        if len(self.cards) != 8 or len(set(self.cards)) != 8:
            raise ValueError(f"{self.name}: a deck is 8 different cards")
        if not set(self.evolutions) <= set(self.cards) or len(self.evolutions) > 2:
            raise ValueError(f"{self.name}: up to 2 evolutions, each a card of the deck")
        if self.flags and len(self.flags) != 8:
            raise ValueError(f"{self.name}: slot flags are one per card")

    @property
    def key(self) -> str:
        """The deck's identity: its cards and evolutions, order-free."""
        evo = "+".join(sorted(self.evolutions))
        return ",".join(sorted(self.cards)) + (f"|evo:{evo}" if evo else "")

    def slots(self, slot_evolution: int) -> list[int]:
        """crforge's slot flags for ``VectorBattle.reset``, one per card in deck order."""
        if self.flags:
            return list(self.flags)
        return [slot_evolution if card in self.evolutions else 0 for card in self.cards]


def is_held_out(deck: Deck | Sequence[str], percent: int = 20) -> bool:
    """Whether a deck belongs to the held-out evaluation set.

    Decided by a hash of the deck's sorted cards alone, so the split is fixed before any training
    and is the same for a hand-picked deck and the same deck met in the replays. About
    ``percent``% of decks are held out. Training never samples a held-out deck as an opponent.
    """
    cards = deck.cards if isinstance(deck, Deck) else tuple(deck)
    digest = hashlib.sha256(",".join(sorted(cards)).encode()).digest()
    return int.from_bytes(digest[:4], "big") % 100 < percent


PIGGIES_EQ = Deck(
    "piggies_eq",
    ("RoyalHogs", "ArcherQueen", "Cannon", "Skeletons", "IceSpirits", "Log", "Earthquake",
     "RoyalDelivery"),
    evolutions=("RoyalHogs", "Cannon"),
    archetype="cycle",
)

_REFERENCE = [
    Deck("hog_26", ("HogRider", "Musketeer", "Cannon", "IceGolemite", "IceSpirits", "Skeletons",
                    "Fireball", "Log"), ("Skeletons", "Cannon"), "cycle"),
    Deck("log_bait", ("GoblinBarrel", "Princess", "GoblinGang", "Knight", "InfernoTower",
                      "Rocket", "Log", "IceSpirits"), ("Knight", "GoblinBarrel"), "bait"),
    Deck("xbow", ("Xbow", "Tesla", "Archer", "Knight", "Skeletons", "IceSpirits", "Fireball",
                  "Log"), ("Knight", "Tesla"), "siege"),
    Deck("mortar_bait", ("Mortar", "Miner", "Bats", "SpearGoblins", "Knight", "Skeletons",
                         "Fireball", "Log"), ("Mortar", "Knight"), "siege"),
    Deck("golem", ("Golem", "DarkWitch", "BabyDragon", "MegaMinion", "RageBarbarian",
                   "Lightning", "Tornado", "Zap"), ("BabyDragon", "RageBarbarian"), "beatdown"),
    Deck("lavaloon", ("LavaHound", "Balloon", "MegaMinion", "Bats", "SkeletonDragons",
                      "Tombstone", "Fireball", "Zap"), ("Bats", "Zap"), "beatdown"),
    Deck("giant_double_prince", ("Giant", "Prince", "DarkPrince", "MegaMinion", "Musketeer",
                                 "Fireball", "Zap", "Bats"), ("Musketeer", "Bats"), "beatdown"),
    Deck("royal_giant", ("RoyalGiant", "Fisherman", "Hunter", "Lightning", "Log", "Skeletons",
                         "ElectroSpirit", "Phoenix"), ("RoyalGiant", "Skeletons"), "beatdown"),
    Deck("pekka_bridge", ("Pekka", "BattleRam", "ElectroWizard", "Ghost", "Assassin", "Poison",
                          "Zap", "Minions"), ("Pekka", "BattleRam"), "bridge_spam"),
    Deck("mega_knight", ("MegaKnight", "Miner", "Bats", "SpearGoblins", "InfernoDragon", "Zap",
                         "Fireball", "Ghost"), ("MegaKnight", "Bats"), "bridge_spam"),
    Deck("miner_control", ("Miner", "Poison", "Valkyrie", "Bats", "Musketeer", "Tesla", "Log",
                           "Skeletons"), ("Valkyrie", "Bats"), "control"),
    Deck("graveyard_freeze", ("Graveyard", "Freeze", "Knight", "IceWizard", "BabyDragon",
                              "Tornado", "Barbarians", "Poison"), ("Knight", "BabyDragon"),
         "control"),
]

REFERENCE_DECKS: dict[str, Deck] = {d.name: d for d in [PIGGIES_EQ, *_REFERENCE]}
