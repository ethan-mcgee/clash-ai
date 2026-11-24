"""RoyaleAPI card keys (``royal-hogs``, ``cannon-ev1``) to crforge card names and slot flags.

RoyaleAPI writes a card in a deck's evolution slot with ``-ev1`` and in its hero slot with
``-hero``. Most names map by dropping the hyphens; the rest are the game's internal codenames,
checked against the tables' text ids (``TID_SPELL_...``).
"""

from __future__ import annotations

# Keys whose crforge name is not the key without hyphens (each checked against its TID).
SPECIAL = {
    "bandit": "Assassin",
    "barbarian-barrel": "BarbLog",
    "cannon-cart": "MovingCannon",
    "dart-goblin": "BlowdartGoblin",
    "elite-barbarians": "AngryBarbarians",
    "executioner": "AxeMan",
    "flying-machine": "DartBarrell",
    "furnace": "FirespiritHut",
    "giant-snowball": "Snowball",
    "guards": "SkeletonWarriors",
    "heal-spirit": "Heal",
    "ice-golem": "IceGolemite",
    "ice-spirit": "IceSpirits",
    "lumberjack": "RageBarbarian",
    "magic-archer": "EliteArcher",
    "mother-witch": "WitchMother",
    "night-witch": "DarkWitch",
    "royal-ghost": "Ghost",
    "rune-giant": "GiantBuffer",
    "skeleton-barrel": "SkeletonBalloon",
    "sparky": "ZapMachine",
    "spirit-empress": "MergeMaiden",
    "the-log": "Log",
    "void": "DarkMagic",
    "zappies": "MiniSparkys",
}

TOWERS = {
    "tower-princess": "King_PrincessTowers",
    "royal-chef": "King_ChefTowers",
    "dagger-duchess": "King_KnifeTowers",
    "cannoneer": "King_CannonTowers",
}

EVOLUTION, HERO = "-ev1", "-hero"


def split_key(key: str) -> tuple[str, str]:
    """``("royal-hogs", "evolution")`` for ``royal-hogs-ev1``; the form is "", "evolution" or
    "hero"."""
    if key.endswith(EVOLUTION):
        return key.removesuffix(EVOLUTION), "evolution"
    if key.endswith(HERO):
        return key.removesuffix(HERO), "hero"
    return key, ""


class CardNames:
    """Maps RoyaleAPI keys to crforge names, checked against a vector's card list."""

    def __init__(self, crforge_cards):
        self.cards = set(crforge_cards)
        # Matched case- and space-blind: crforge names one card "Elixir Collector".
        self._lower = {c.lower().replace(" ", ""): c for c in self.cards}

    def name(self, key: str) -> str:
        base, _ = split_key(key)
        if base in SPECIAL:
            return SPECIAL[base]
        plain = base.replace("-", "")
        for candidate in (plain, plain + "s", plain.removesuffix("s")):
            if candidate in self._lower:
                return self._lower[candidate]
        raise KeyError(f"no crforge card for RoyaleAPI key {key!r}")

    def deck(self, keys) -> tuple[list[str], list[str]]:
        """A deck's crforge names, and the form of each slot ("", "evolution" or "hero")."""
        names, forms = [], []
        for key in keys:
            names.append(self.name(key))
            forms.append(split_key(key)[1])
        return names, forms
