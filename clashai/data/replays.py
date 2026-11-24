"""Reading the replay dataset's Parquet shards into plain records."""

from __future__ import annotations

import glob
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq


@dataclass(frozen=True)
class Player:
    deck: tuple[str, ...]
    """RoyaleAPI card keys in recorded order, with ``-ev1`` / ``-hero`` suffixes."""
    levels: tuple[int, ...]
    tower: str
    tower_level: int
    final_tower_hp: dict


@dataclass(frozen=True)
class Replay:
    tag: str
    league: str
    mode: str
    team: Player
    opponent: Player
    result: str
    """``victory``, ``defeat`` or ``draw``, from the team's side."""
    team_crowns: int
    opponent_crowns: int
    payload: dict

    def side(self, name: str) -> Player:
        return self.team if name == "team" else self.opponent

    def won(self, name: str) -> bool | None:
        if self.result == "draw":
            return None
        return (self.result == "victory") == (name == "team")


def _player(raw: dict) -> Player:
    p = raw["players"][0]
    return Player(tuple(c["card_key"] for c in p["deck"]), tuple(c["level"] for c in p["deck"]),
                  p["tower_card"]["card_key"], p["tower_card"]["level"],
                  p.get("source_final_tower_hp", {}))


def read_replays(root: str | Path) -> Iterator[Replay]:
    """Every replay of the dataset at ``root``, shard by shard."""
    for shard in sorted(glob.glob(str(Path(root) / "replays" / "*.parquet"))):
        table = pq.read_table(shard, columns=["replay_tag", "ranked_league", "game_mode",
                                              "payload_json"])
        for tag, league, mode, payload in zip(*(table.column(i).to_pylist() for i in range(4)),
                                              strict=True):
            p = json.loads(payload)
            b = p["battle"]
            yield Replay(tag, league or "", mode, _player(b["team"]), _player(b["opponent"]),
                         b["result"], b["team"]["crowns"], b["opponent"]["crowns"], p)
