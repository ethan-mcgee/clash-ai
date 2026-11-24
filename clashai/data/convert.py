"""RoyaleAPI replays into crforge battles.

    python -m clashai.data.convert --deck piggies_eq --limit 200

The dataset records each battle's decks, levels, towers and every play (card, point, 20 Hz tick),
but not the battle's random seed or either player's deck order. For each replay:

1. **Deck order.** Each side's plays fix which opening hands and queue orders are possible
   (:mod:`.cycle`). One is picked at random among them.
2. **Deal.** crforge deals ``DeckShuffle.order(deck, draw + accountID.lo)`` (:mod:`.shuffle`). A
   shuffle seed giving the picked order is looked up, and each side's ``accountID.lo`` is set to
   reach it from the battle seed's draw. The battle seed itself is a hash of the replay's tag: the
   real one is unknown, so random effects in the battle may differ from the recording.
3. **Drive.** The battle is started from a replay document holding that setup and stepped tick by
   tick. Each recorded play is given on its recorded tick as the action for the tile holding its
   point, in the player's frame. A play the engine cannot take yet (elixir short, card not yet back
   in hand, tile closed) is retried each tick for up to :data:`MAX_DELAY` ticks, then dropped.
4. **Check.** At the battle's end its crowns and tower hit points are compared with the recording.

Each battle is written as a crforge replay document, which crforge's replay labeller turns into
(observation, mask, action) rows. A summary line per replay records how faithful it was.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .. import paths
from ..sim.env import DEFAULT_LEVEL
from .cards import TOWERS, CardNames, split_key
from .cycle import feasible_keys, key_of
from .download import dataset_dir
from .replays import Replay, read_replays
from .shuffle import OMIT_FROM_STARTING_HAND, _i32, deal_order, shuffle_draws

MAX_DELAY = 20
"""Ticks a play may run late before it is dropped (one second)."""
WIDTH, HEIGHT, TILE = 18000, 32000, 1000
ARENA = "Arena_L"
"""The arena both avatars name in written documents; it sets nothing in the battle."""


# -- events ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    side: str  # "team" or "opponent"
    kind: str  # "play_card" or "activate_ability"
    tick: int
    card: str | None  # RoyaleAPI key
    x: int | None  # native world units, team at the bottom
    y: int | None
    data_i: int


def load_events(root: Path, tags: set[str]) -> dict[str, list[Event]]:
    """The recorded events of the replays ``tags``, in source order."""
    out: dict[str, list[Event]] = defaultdict(list)
    cols = ["replay_tag", "source_index", "side", "kind", "replay_tick_20hz", "card_key",
            "native_x", "native_y", "data_i"]
    wanted = pa.array(sorted(tags))
    for shard in sorted((root / "actions").glob("*.parquet")):
        t = pq.read_table(shard, columns=cols)
        t = t.filter(pc.is_in(t["replay_tag"], value_set=wanted))
        if t.num_rows == 0:
            continue
        t = t.sort_by([("replay_tag", "ascending"), ("source_index", "ascending")])
        d = t.to_pydict()
        for i in range(t.num_rows):
            out[d["replay_tag"][i]].append(Event(
                d["side"][i], d["kind"][i], d["replay_tick_20hz"][i], d["card_key"][i],
                d["native_x"][i], d["native_y"][i], d["data_i"][i]))
    return out


# -- the deal ------------------------------------------------------------------------------------

class SeedTable:
    """A shuffle seed for every deck order of a deck with no held-back card.

    Such a deck's dealt order depends on the seed alone, not on its cards, so one table serves
    every deck.
    """

    def __init__(self, size: int = 60_000):
        cache = paths.data_root() / "cache" / f"seed_table_{size}.json"
        if cache.is_file():
            raw = json.loads(cache.read_text())
        else:
            names = [f"c{i}" for i in range(8)]
            raw = {}
            for seed in range(size):
                order = deal_order(names, seed)
                raw.setdefault(_key_text(key_of(order)), seed)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(raw))
        self.seeds = raw

    def seed_for(self, deck: list[str], key) -> int | None:
        if not OMIT_FROM_STARTING_HAND & set(deck):
            return self.seeds.get(_key_text(key))
        for seed in range(200_000):  # a held-back card changes the deal: search this deck's own
            if key_of(deal_order(deck, seed)) == key:
                return seed
        return None


def _key_text(key) -> str:
    return ",".join(map(str, sorted(key[0]))) + "|" + ",".join(map(str, key[1]))


# -- one replay's plan ---------------------------------------------------------------------------

@dataclass
class Plan:
    tag: str
    sides: tuple[str, str]  # which recorded side is crforge side 0 and side 1
    decks: list[list[str]]
    slots: list[list[int]]
    levels: list[list[int]]
    towers: list[str]
    tower_levels: list[int]
    rnd_seed: int
    words: list[int]
    events: list[tuple[int, int, str, int | None, int | None, int | None]]
    """(tick, crforge side, kind, deck index or None, world x, world y), in recorded order."""
    feasible: list[int]  # how many deck orders fitted each side
    recorded: dict
    problem: str = ""
    keys: list = field(default_factory=list)  # the deck order planned for each side


def _world(x: int, y: int, mapping: str, team_side: int, is_team: bool) -> tuple[int, int]:
    """A recorded point (team at the bottom) in crforge's world units."""
    if mapping == "native":
        return x, y
    if mapping == "screen":  # the team player's own screen: side 0's frame mirrors x, side 1's y
        return (WIDTH - x, y) if team_side == 0 else (x, HEIGHT - y)
    if mapping == "rotate":  # the recording shows the team's side at the bottom of a 180° turn
        bottom_is_side0 = team_side == 0
        return (x, y) if bottom_is_side0 else (WIDTH - x, HEIGHT - y)
    raise ValueError(mapping)


def plan_replay(r: Replay, events: list[Event], names: CardNames, table: SeedTable,
                rng: np.random.Generator, mapping: str = "native",
                sides_by: str = "team0", seed_variant: int = 0, tick_offset: int = 0) -> Plan:
    team_side = 0
    if sides_by == "data_i":
        team_side = next((e.data_i for e in events if e.side == "team"), 0)
    sides = ("team", "opponent") if team_side == 0 else ("opponent", "team")
    decks, slots, levels, towers, tower_levels = [], [], [], [], []
    for name in sides:
        p = r.side(name)
        cards, forms = names.deck(p.deck)
        decks.append(cards)
        slots.append([{"": 0, "evolution": 1, "hero": 2}[f] for f in forms])
        levels.append(list(p.levels))
        towers.append(TOWERS[p.tower])
        tower_levels.append(p.tower_level)
    recorded = {"result": r.result, "crowns": {"team": r.team_crowns, "opponent": r.opponent_crowns},
                "towers": {"team": r.team.final_tower_hp, "opponent": r.opponent.final_tower_hp},
                "duration_s": r.payload.get("replay", {}).get("duration", {}).get("timeline_seconds")}
    plan_events, plays = [], [[], []]
    base_index = [{split_key(k)[0]: i for i, k in enumerate(r.side(n).deck)} for n in sides]
    for e in events:
        side = sides.index(e.side)
        if e.kind == "play_card":
            index = base_index[side].get(split_key(e.card)[0]) if e.card else None
            if index is None:
                return Plan(r.tag, sides, decks, slots, levels, towers, tower_levels, 0, [0, 0],
                            [], [0, 0], recorded, f"play of a card not in the deck: {e.card}")
            wx, wy = _world(e.x, e.y, mapping, team_side, e.side == "team")
            plan_events.append((e.tick + tick_offset, side, "play", index, wx, wy))
            plays[side].append(index)
        elif e.kind == "activate_ability":
            plan_events.append((e.tick + tick_offset, side, "ability", None, None, None))
    salt = r.tag if seed_variant == 0 else f"{r.tag}:{seed_variant}"
    rnd_seed = _i32(int.from_bytes(hashlib.sha256(salt.encode()).digest()[:4], "big"))
    draws = shuffle_draws(rnd_seed)
    words, counts, chosen = [], [], []
    for side in (0, 1):
        held = {i for i, c in enumerate(decks[side]) if c in OMIT_FROM_STARTING_HAND}
        keys = [k for k in feasible_keys(plays[side]) if not held & k[0]]  # never dealt in hand
        counts.append(len(keys))
        if not keys:
            return Plan(r.tag, sides, decks, slots, levels, towers, tower_levels, rnd_seed,
                        [0, 0], plan_events, counts, recorded,
                        f"no deck order explains side {side}'s plays")
        key = keys[rng.integers(len(keys))]
        chosen.append(key)
        seed = table.seed_for(decks[side], key)
        if seed is None:
            return Plan(r.tag, sides, decks, slots, levels, towers, tower_levels, rnd_seed,
                        [0, 0], plan_events, counts, recorded, "no shuffle seed for the order")
        words.append(_i32(seed - draws[side]))
    return Plan(r.tag, sides, decks, slots, levels, towers, tower_levels, rnd_seed, words,
                plan_events, counts, recorded, keys=chosen)


# -- driving -------------------------------------------------------------------------------------

@dataclass
class Outcome:
    tag: str
    problem: str = ""
    plays: int = 0
    abilities: int = 0
    on_time: int = 0
    late: int = 0
    dropped: int = 0
    max_delay: int = 0
    exact_points: int = 0
    end_tick: int = 0
    crowns: dict = field(default_factory=dict)
    towers: dict = field(default_factory=dict)
    crowns_match: bool = False
    result_match: bool = False
    towers_exact: bool = False
    tower_error: int = 0
    feasible_orders: list = field(default_factory=list)
    drops: list = field(default_factory=list)
    deal_ok: bool = False
    misses: list = field(default_factory=list)  # [side, card, dx, dy] of plays off their point
    not_modelled: int = 0  # commands crforge does not model (counted by the engine)
    fault: str = "none"
    recorded_s: float | None = None
    sides: list = field(default_factory=list)  # recorded side of crforge side 0, then side 1
    decks: list = field(default_factory=list)  # crforge deck of each side
    slots: list = field(default_factory=list)


def _tile(side: int, x: int, y: int) -> tuple[int, int]:
    """The tile of a seat's frame holding a world point (measured: side 0's frame mirrors x,
    side 1's mirrors y)."""
    tx, ty = min(max(x // TILE, 0), 17), min(max(y // TILE, 0), 31)
    return (17 - tx, ty) if side == 0 else (tx, 31 - ty)


def drive(plans: list[Plan], vector, out_dir: Path | None) -> list[Outcome]:
    """Drives one batch of planned replays, one per environment of ``vector`` (1-tick decisions)."""
    s, h = vector.schema, vector.schema.header
    n = len(plans)
    from crforge_env import SideSetup

    setups = [[SideSetup(towers=p.towers[side], king_level=p.tower_levels[side],
                         tower_level=p.tower_levels[side], card_levels=p.levels[side])
               for p in plans] for side in (0, 1)]
    pad = vector.count - n
    filler = plans[0]
    vector.reset([0] * vector.count,
                 [p.decks[0] for p in plans] + [filler.decks[0]] * pad,
                 [p.decks[1] for p in plans] + [filler.decks[1]] * pad,
                 [p.slots[0] for p in plans] + [filler.slots[0]] * pad,
                 [p.slots[1] for p in plans] + [filler.slots[1]] * pad,
                 setups[0] + [setups[0][0]] * pad, setups[1] + [setups[1][0]] * pad)
    vector.actions[:] = s.wait
    vector.step()
    docs = []
    for i, p in enumerate(plans):
        doc = json.loads(vector.replay_json(i))
        doc["rndSeed"] = p.rnd_seed
        for side in (0, 1):
            doc["battle"][f"avatar{side}"]["accountID.lo"] = p.words[side]
        doc["cmd"] = []
        docs.append(doc)
    vector.start_from_replays(list(range(n)), docs, [0] * n)

    ids = [[[vector.card_id(c) for c in p.decks[side]] for side in (0, 1)] for p in plans]
    deal_ok = []
    for i, p in enumerate(plans):
        ok = True
        for side in (0, 1):
            obs = vector.observations[i, side]
            index = {card: k for k, card in enumerate(ids[i][side])}
            hand = frozenset(index[int(obs[j])] for j in h["hand_card"])
            queue = tuple(index[int(obs[j])] for j in h["queue"])
            ok &= (hand, queue) == p.keys[side]
        deal_ok.append(ok)
    queues = [[[e for e in p.events if e[1] == side] for side in (0, 1)] for p in plans]
    heads = [[0, 0] for _ in plans]
    waiting = [[0, 0] for _ in plans]  # ticks the head event has waited
    outcomes = [Outcome(p.tag, plays=sum(e[2] == "play" for e in p.events),
                        abilities=sum(e[2] == "ability" for e in p.events),
                        feasible_orders=p.feasible, deal_ok=ok, sides=list(p.sides),
                        decks=p.decks, slots=p.slots)
                for p, ok in zip(plans, deal_ok, strict=True)]
    finished = np.zeros(vector.count, bool)
    finished[n:] = True
    tiles = s.cols * s.rows
    never = np.iinfo(np.int64).max
    # Each side's next command tick, so a tick with nothing due costs one comparison.
    due = np.full((vector.count, 2), never, np.int64)
    for i in range(n):
        for side in (0, 1):
            if queues[i][side]:
                due[i, side] = queues[i][side][0][0]
    tick_index = h["tick"][0]
    while not finished.all():
        actions = np.full((vector.count, 2), s.wait, np.int32)
        ticks = vector.observations[:, 0, tick_index].astype(np.int64)
        for i, side in zip(*np.nonzero((due <= ticks[:, None]) & ~finished[:, None]),
                           strict=True):
            i, side = int(i), int(side)
            q, k = queues[i][side], heads[i][side]
            _, _, kind, index, wx, wy = q[k]
            obs, mask = vector.observations[i, side], vector.masks[i, side]
            action = None
            if kind == "ability":
                open_ = np.flatnonzero(mask[s.first_ability:])
                if len(open_):
                    action = s.first_ability + int(open_[0])
            else:
                hand = [int(obs[j]) for j in h["hand_card"]]
                card = ids[i][side][index]
                if card in hand:
                    col, row = _tile(side, wx, wy)
                    a = 1 + hand.index(card) * tiles + row * s.cols + col
                    if mask[a]:
                        action = a
            if action is not None:
                actions[i, side] = action
                delay = waiting[i][side]
                out = outcomes[i]
                out.on_time += delay == 0
                out.late += delay > 0
                out.max_delay = max(out.max_delay, delay)
                heads[i][side] += 1
                waiting[i][side] = 0
                due[i, side] = q[k + 1][0] if k + 1 < len(q) else never
            elif waiting[i][side] >= MAX_DELAY:
                outcomes[i].dropped += 1
                outcomes[i].drops.append([q[k][0], side, kind,
                                          plans[i].decks[side][index] if index is not None
                                          else None])
                heads[i][side] += 1
                waiting[i][side] = 0
                due[i, side] = q[k + 1][0] if k + 1 < len(q) else never
            else:
                waiting[i][side] += 1
        vector.actions[:] = actions
        vector.step()
        for i in np.flatnonzero(vector.infos[:n, s.info["not_modelled"]]):
            outcomes[i].not_modelled += int(vector.infos[i, s.info["not_modelled"]])
        for i in np.flatnonzero(vector.dones.astype(bool) & ~finished):
            finished[i] = True
            _finish(plans[i], outcomes[i], vector, i, out_dir)
    return outcomes


def _finish(p: Plan, out: Outcome, vector, i: int, out_dir: Path | None) -> None:
    s, h = vector.schema, vector.schema.header
    info = vector.infos[i]
    out.end_tick = int(info[s.info["tick"]])
    out.fault = s.fault_names[int(info[s.info["fault"]])]
    out.recorded_s = p.recorded.get("duration_s")
    crowns = {p.sides[side]: int(info[s.info[f"crowns_{side}"]]) for side in (0, 1)}
    out.crowns = crowns
    out.crowns_match = crowns == p.recorded["crowns"]
    winner = int(info[s.info["winner"]])
    result = ("draw" if winner == s.winner_draw else
              "victory" if winner in (0, 1) and p.sides[winner] == "team" else "defeat")
    out.result_match = result == p.recorded["result"]
    o = vector.observations[i, 0]  # side 0's view: own towers then the enemy's
    hp = {p.sides[0]: [int(o[h[f"own_{t}_hp"][0]]) for t in ("king", "left", "right")],
          p.sides[1]: [int(o[h[f"enemy_{t}_hp"][0]]) for t in ("king", "left", "right")]}
    out.towers = hp
    err = 0
    for name, (king, a, b) in hp.items():
        rec = p.recorded["towers"].get(name, {})
        if not rec:
            continue
        rk, r0, r1 = rec.get("king", 0), rec.get("princess0", 0), rec.get("princess1", 0)
        # Which princess the recording calls 0 is not documented: take the closer pairing.
        err += abs(king - rk) + min(abs(a - r0) + abs(b - r1), abs(a - r1) + abs(b - r0))
    out.tower_error = err
    out.towers_exact = err == 0
    doc = json.loads(vector.replay_json(i))
    if out_dir is not None:
        recorded = {(c["c"]["t"], c["c"]["idHi"]): (c["c"]["px"], c["c"]["py"])
                    for c in doc["cmd"] if "px" in c["c"]}
        for tick, side, kind, index, wx, wy in p.events:
            got = recorded.get((tick, side))
            if kind != "play" or got is None:
                continue
            # crforge snaps a troop's point 1 unit (the game's symmetrical snap); the recording
            # does not. Within 1 unit is the same tile and the same placement.
            if abs(got[0] - wx) <= 1 and abs(got[1] - wy) <= 1:
                out.exact_points += 1
            else:
                out.misses.append([side, p.decks[side][index], got[0] - wx, got[1] - wy])
        (out_dir / "docs").mkdir(parents=True, exist_ok=True)
        (out_dir / "docs" / f"{p.tag}.json").write_text(json.dumps(doc), encoding="utf-8")


# -- the run -------------------------------------------------------------------------------------

def is_deck(r: Replay, side: str, keys: set[str], evolutions: set[str] | None) -> bool:
    deck = r.side(side).deck
    if {split_key(k)[0] for k in deck} != keys:
        return False
    return evolutions is None or {k for k in deck if split_key(k)[1]} == evolutions


PIGGIES = ({"royal-hogs", "archer-queen", "cannon", "skeletons", "ice-spirit", "the-log",
            "earthquake", "royal-delivery"}, {"royal-hogs-ev1", "cannon-ev1"})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert RoyaleAPI replays into crforge battles")
    parser.add_argument("--deck", choices=["piggies_eq", "all"], default="piggies_eq")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--mapping", choices=["native", "screen", "rotate"], default="screen")
    parser.add_argument("--sides", choices=["team0", "data_i"], default="team0")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0, help="picks among fitting deck orders")
    parser.add_argument("--seed-variant", type=int, default=0, help="changes the battle seed")
    parser.add_argument("--tick-offset", type=int, default=0, help="added to every recorded tick")
    args = parser.parse_args(argv)

    from ..sim.env import make_vector
    from crforge_env import LadderConfig, VectorBattle

    out_dir = args.out or paths.data_root() / "converted" / f"{args.deck}-{args.mapping}-{args.sides}-v{args.seed_variant}-t{args.tick_offset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    root = dataset_dir()
    replays = []
    for r in read_replays(root):
        if args.deck == "all" or any(is_deck(r, side, *PIGGIES) for side in ("team", "opponent")):
            replays.append(r)
            if args.limit and len(replays) >= args.limit:
                break
    print(f"{len(replays)} replays selected", flush=True)
    events = load_events(root, {r.tag for r in replays})
    paths.ensure_java()
    vector = VectorBattle(min(args.batch, len(replays)), paths.default_threads(),
                          install=paths.install_dir(), tables=paths.tables_dir(),
                          config=LadderConfig(decision_ticks=1, level=DEFAULT_LEVEL,
                                              tower_level=DEFAULT_LEVEL, arena=ARENA))
    names = CardNames(vector.vocabulary[i] for i in vector.card_types())
    table = SeedTable()
    rng = np.random.default_rng(args.seed)
    summary_path = out_dir / "summary.jsonl"
    t0 = time.time()
    done = 0
    with open(summary_path, "w", encoding="utf-8") as summary:
        pending: list[Plan] = []

        def flush():
            nonlocal done
            if pending:
                for o in drive(pending, vector, out_dir):
                    summary.write(json.dumps(asdict(o)) + "\n")
                done += len(pending)
                pending.clear()
                print(f"{done}/{len(replays)} replays, {time.time() - t0:.0f} s", flush=True)

        for r in replays:
            plan = plan_replay(r, events.get(r.tag, []), names, table, rng, args.mapping, args.sides,
                               args.seed_variant, args.tick_offset)
            if plan.problem:
                summary.write(json.dumps(asdict(Outcome(plan.tag, problem=plan.problem,
                                                        feasible_orders=plan.feasible))) + "\n")
                continue
            pending.append(plan)
            if len(pending) == vector.count:
                flush()
        flush()
    print(f"written to {out_dir}")
    report(summary_path)


def report(summary_path: Path) -> None:
    rows = [json.loads(line) for line in open(summary_path, encoding="utf-8")]
    ok = [r for r in rows if not r["problem"]]
    plays = sum(r["plays"] + r["abilities"] for r in ok)
    print(f"\n{len(rows)} replays: {len(rows) - len(ok)} not convertible "
          f"({dict(_count(r['problem'].split(':')[0] for r in rows if r['problem']))})")
    if not ok:
        return
    print(f"deals as planned: {sum(r['deal_ok'] for r in ok) / len(ok):.1%}")
    print(f"commands: {sum(r['on_time'] for r in ok) / plays:.1%} on time, "
          f"{sum(r['late'] for r in ok) / plays:.1%} late, {sum(r['dropped'] for r in ok) / plays:.1%} dropped")
    played = sum(r["plays"] for r in ok)
    print(f"plays landing on the recorded point (within the 1-unit snap): {sum(r['exact_points'] for r in ok) / played:.1%}")
    print(f"replays with no dropped command: {sum(r['dropped'] == 0 for r in ok) / len(ok):.1%}")
    print(f"result matches: {sum(r['result_match'] for r in ok) / len(ok):.1%}; crowns match: "
          f"{sum(r['crowns_match'] for r in ok) / len(ok):.1%}; tower HP exact: "
          f"{sum(r['towers_exact'] for r in ok) / len(ok):.1%}")
    err = np.array([r["tower_error"] for r in ok])
    print(f"tower HP error (sum over 6 towers): median {np.median(err):.0f}, "
          f"75th pct {np.percentile(err, 75):.0f}")


def _count(items):
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


if __name__ == "__main__":
    main()
