"""A learned world model and a model-based planner that sees only the text observations.

The oracle reads the true map and the LLM is told the rules in its prompt. This agent gets
neither. It has to work out what tiles and actions do from experience, then plan with what
it learned. It has three parts:

1. **Perception -> belief map.** Each text observation is parsed into absolute map cells:
   the four neighbours and the agent's own square (always identified), objects within
   ``view_radius`` (identified), and every other cell in that window, which is known only
   to be *wall or floor* (the observation lists only interesting tiles). Cells never in view
   are unknown.

2. **Learned transition model.** Every step is logged as
   ``(action kind, tile here, tile ahead, items held) -> effect``. An effect is a symbolic
   diff: did the agent move, did the tile here or ahead change into something else, what
   was gained or lost, and which *global* tile changes happened ("every closed gate became
   an open gate"). For each action kind the model fits a small decision tree over the three
   context features. It splits only where the data demands it, so it learns *which features
   matter* (moving depends only on the tile ahead, ``pick_up`` only on the tile here) and
   generalises to contexts it has never seen.

   Effects are scored by *consistency*, not frequency. An observation can't refute
   "pressing the switch opens every gate" if no closed gate was in view. If a change is
   only noticed later (a gate seen closed, then seen open after the agent walked back),
   it is credited back to the earlier action that seemed to do nothing.

3. **Planning in imagination.** Dijkstra search over imagined states (position, facing,
   items, tiles changed, global changes applied), with the learned model as the simulator.
   Unknown cells are assumed walkable but cost extra, which makes the agent optimistic
   about unexplored space. If the goal is not on the map yet, it explores: it heads for the
   imagined state with the best information gain per step (unseen cells it would bring into
   view, unidentified neighbours, and optionally untried action contexts as curiosity). It
   replans after every step, so a wrong prediction ("that was a wall, not floor") is fixed
   on the next step.

The model is pretrained by curiosity-driven exploration of separate training maps (none of
the evaluation tasks). Each evaluation episode then starts from a copy of that pretrained
model and keeps learning within the episode. That way episodes stay independent and
comparable across conditions.

    python -m gridworld_planner.worldmodel                 # pretrain, print the learned rules + accuracy
    python -m gridworld_planner.worldmodel --task two_room_chain   # watch it solve a task
"""
from __future__ import annotations

import argparse
import copy
import heapq
import itertools
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from .env import ACTIONS, DEFAULT_LAYOUT, DIRS, NAMES, GridWorld

PLAIN = "wall or floor"          # in view but not listed: the observation says only "not interesting"
FEATURES = ("here", "target", "holding")
UNCERTAIN_STEP_COST = 0.5        # extra cost for stepping into a cell the agent hasn't identified
UNIDENTIFIED_WEIGHT = 0.1        # info value of a wall-or-floor neighbour vs. a never-seen cell (1.0)

# Separate training maps. None of these is an evaluation task layout.
TRAINING_LAYOUTS = [
    DEFAULT_LAYOUT,
    """
#########
#@.k#...#
#.s.D.#.#
#...#.X.#
#######G#
#########
""",
    """
########
#s.X..G#
#.######
#k.D.@.#
########
""",
]


# --------------------------------------------------------------------------- perception
@dataclass
class Observation:
    pos: tuple[int, int]
    facing: str
    holding: tuple[str, ...]
    here: str
    around: dict[str, str]
    nearby: list[tuple[str, tuple[int, int]]]
    message: str


_POS_RE = re.compile(r"You are at row (-?\d+), column (-?\d+), facing (\w+)\.")
_ITEM_RE = re.compile(r"(.+?) \((.+)\)$")


def _parse_offset(text: str) -> tuple[int, int]:
    dr = dc = 0
    for part in text.split(", "):
        n, d = part.split()
        if d in ("up", "down"):
            dr = int(n) * (1 if d == "down" else -1)
        else:
            dc = int(n) * (1 if d == "right" else -1)
    return dr, dc


def parse_observation(text: str) -> Observation:
    """Turn the env's text observation into structured data (the only input the agent gets)."""
    message, pos, facing, holding, here, around, nearby = [], None, "up", (), "floor", {}, []
    for line in text.splitlines():
        if m := _POS_RE.search(line):
            pos, facing = (int(m[1]), int(m[2])), m[3]
        elif line.startswith("Holding: "):
            body = line[len("Holding: "):].rstrip(".")
            holding = () if body == "nothing" else tuple(sorted(body.split(", ")))
        elif line.startswith("Here: "):
            here_part, around_part = line.split(". Around you: ")
            here = here_part[len("Here: "):]
            for item in around_part.rstrip(".").split(", "):
                d, name = item.split(": ")
                around[d] = name
        elif line.startswith("Nearby: "):
            body = line[len("Nearby: "):].rstrip(".")
            if body != "nothing of interest":
                for item in body.split("; "):
                    m = _ITEM_RE.match(item)
                    nearby.append((m[1], _parse_offset(m[2])))
        elif pos is None:
            message.append(line)
    if pos is None:
        raise ValueError("observation has no position line")
    return Observation(pos, facing, holding, here, around, nearby, " ".join(message))


class Belief:
    """What the agent believes about the map: identified tiles, PLAIN cells, and when each was last seen."""

    def __init__(self, radius: int = 2):
        self.radius = radius
        self.tiles: dict[tuple[int, int], str] = {}
        self.last_seen: dict[tuple[int, int], int] = {}
        self.t = -1
        self.pos = (0, 0)
        self.facing = "up"
        self.holding: tuple[str, ...] = ()

    def window(self, pos):
        r, c = pos
        k = self.radius
        return [(rr, cc) for rr in range(r - k, r + k + 1) for cc in range(c - k, c + k + 1)]

    def update(self, obs: Observation) -> list[tuple[tuple[int, int], str, str, int]]:
        """Fold an observation in. Returns changes between identified tiles: (cell, old, new, last seen before)."""
        self.t += 1
        self.pos, self.facing, self.holding = obs.pos, obs.facing, obs.holding
        r, c = obs.pos
        seen = {obs.pos: obs.here}
        for d, name in obs.around.items():
            dr, dc = DIRS[d]
            seen[(r + dr, c + dc)] = name
        for name, (dr, dc) in obs.nearby:
            seen[(r + dr, c + dc)] = name
        changes = []
        for cell in self.window(obs.pos):
            old = self.tiles.get(cell)
            new = seen.get(cell, PLAIN)
            if new == PLAIN and old in ("wall", "floor"):
                new = old  # consistent with what we already know
            if old not in (None, PLAIN) and new != PLAIN and new != old:
                changes.append((cell, old, new, self.last_seen.get(cell, -1)))
            self.tiles[cell] = new
            self.last_seen[cell] = self.t
        return changes

    def visible_tiles(self) -> frozenset[str]:
        return frozenset(self.tiles[c] for c in self.window(self.pos) if self.tiles.get(c, PLAIN) != PLAIN)

    def bounds(self):
        rows = [p[0] for p in self.tiles]
        cols = [p[1] for p in self.tiles]
        return min(rows) - 1, max(rows) + 1, min(cols) - 1, max(cols) + 1

    def render(self) -> str:
        """ASCII picture of the belief: '?' never seen, ',' wall-or-floor, '@' the agent."""
        char = {name: ch for ch, name in NAMES.items()} | {PLAIN: ","}
        r0, r1, c0, c1 = self.bounds()
        return "\n".join("".join("@" if (r, c) == self.pos else char.get(self.tiles.get((r, c)), "?")
                                 for c in range(c0 + 1, c1)) for r in range(r0 + 1, r1))


# --------------------------------------------------------------------------- learned dynamics
@dataclass(frozen=True)
class Effect:
    moved: bool = False
    here_to: str | None = None
    target_to: str | None = None
    gain: tuple[str, ...] = ()
    lose: tuple[str, ...] = ()
    glob: frozenset = frozenset()   # {(from_tile, to_tile)}: every such tile changes

    @property
    def local(self):
        return self.moved, self.here_to, self.target_to, self.gain, self.lose

    @property
    def content(self) -> int:
        return sum(map(bool, self.local)) + len(self.glob)

    def describe(self) -> str:
        parts = ["move" if self.moved else ""]
        if self.here_to:
            parts.append(f"here becomes {self.here_to}")
        if self.target_to:
            parts.append(f"ahead becomes {self.target_to}")
        parts += [f"gain {g}" for g in self.gain] + [f"lose {item}" for item in self.lose]
        parts += [f"every {a} becomes {b}" for a, b in sorted(self.glob)]
        return ", ".join(p for p in parts if p) or "nothing happens"


@dataclass
class Record:
    """One observed transition. ``glob`` is mutable so later evidence can be credited back to it."""
    kind: str
    feats: tuple[str, str, str]
    local: tuple
    glob: set
    visible_after: frozenset

    @property
    def effect(self) -> Effect:
        moved, here_to, target_to, gain, lose = self.local
        return Effect(moved, here_to, target_to, gain, lose, frozenset(self.glob))


def consistent(h: Effect, rec: Record) -> bool:
    """Could hypothesis h have produced what was seen? Unseen global changes can't refute it."""
    if h.local != rec.local or not rec.glob <= h.glob:
        return False
    return all(a not in rec.visible_after for a, _ in h.glob - rec.glob)


@dataclass
class Node:
    best: Effect
    n: int
    explained: int
    feature: int | None = None
    children: dict = field(default_factory=dict)


class WorldModel:
    """Per action kind, a decision tree over (here, target, holding) whose leaves are effects."""

    def __init__(self):
        self.records: dict[str, list[Record]] = {}
        self.trees: dict[str, Node] = {}
        self.seen: set[tuple[str, tuple]] = set()
        self._dirty = True
        self._cache: dict = {}

    def add(self, rec: Record) -> None:
        self.records.setdefault(rec.kind, []).append(rec)
        self.seen.add((rec.kind, rec.feats))
        self.mark_dirty()

    def mark_dirty(self):
        self._dirty, self._cache = True, {}

    def __len__(self):
        return sum(map(len, self.records.values()))

    # fitting
    def fit(self) -> None:
        self.trees = {}
        for kind, recs in self.records.items():
            hyps = list(dict.fromkeys(r.effect for r in recs))
            cons = [frozenset(i for i, h in enumerate(hyps) if consistent(h, r)) for r in recs]
            exact = [hyps.index(r.effect) for r in recs]
            self.trees[kind] = self._build(list(range(len(recs))), [0, 1, 2], recs, hyps, cons, exact)
        self._dirty = False

    @staticmethod
    def _best(idx, hyps, cons, exact):
        count = Counter(h for i in idx for h in cons[i])
        support = Counter(exact[i] for i in idx)
        h = max(range(len(hyps)), key=lambda j: (count[j], support[j], hyps[j].content))
        return h, count[h]

    def _build(self, idx, feats_left, recs, hyps, cons, exact) -> Node:
        h, explained = self._best(idx, hyps, cons, exact)
        node = Node(hyps[h], len(idx), explained)
        if explained == len(idx) or not feats_left:
            return node
        best_split = None
        for f in feats_left:
            groups: dict[str, list[int]] = {}
            for i in idx:
                groups.setdefault(recs[i].feats[f], []).append(i)
            scores = [self._best(g, hyps, cons, exact)[1] for g in groups.values()]
            pure = sum(s == len(g) for s, g in zip(scores, groups.values()))
            key = (sum(scores), pure, -len(groups))
            if best_split is None or key > best_split[0]:
                best_split = (key, f, groups)
        _, f, groups = best_split
        node.feature = f
        rest = [x for x in feats_left if x != f]
        node.children = {v: self._build(g, rest, recs, hyps, cons, exact) for v, g in groups.items()}
        return node

    # querying
    def predict(self, kind: str, feats: tuple) -> tuple[Effect | None, bool, bool]:
        """Returns (effect, unknown, novel).

        ``effect`` is the predicted effect, or None if this action kind has never been tried.
        ``unknown`` means the tree hit a feature value it has never seen (e.g. a new tile
        type ahead), so the prediction is a guess. ``novel`` means this exact context has
        never been tried, even if the tree generalises to it confidently.
        """
        key = (kind, feats)
        if key in self._cache:
            return self._cache[key]
        if self._dirty:
            self.fit()
        node, unknown = self.trees.get(kind), kind not in self.trees
        while node is not None and node.feature is not None:
            child = node.children.get(feats[node.feature])
            if child is None:
                unknown = True  # unseen value: fall back to this node's best effect
                break
            node = child
        out = (node.best if node else None, unknown, key not in self.seen)
        self._cache[key] = out
        return out

    def describe(self) -> str:
        if self._dirty:
            self.fit()
        lines = []

        def walk(node, depth, label):
            pad = "  " * depth
            if node.feature is None:
                conf = f"{node.explained}/{node.n}"
                lines.append(f"{pad}{label}-> {node.best.describe()}   [{conf} transitions]")
                return
            lines.append(f"{pad}{label}split on {FEATURES[node.feature]}  (otherwise: {node.best.describe()})")
            for v, child in sorted(node.children.items()):
                walk(child, depth + 1, f"{FEATURES[node.feature]} = {v}: ")

        for kind in ("move", "pick_up", "use"):
            if kind in self.trees:
                lines.append(f"{kind}:")
                walk(self.trees[kind], 1, "")
        return "\n".join(lines)


# --------------------------------------------------------------------------- planning
@dataclass(frozen=True)
class SimState:
    pos: tuple[int, int]
    facing: str
    holding: tuple[str, ...]
    overrides: frozenset = frozenset()   # {(cell, tile)} changed in imagination
    glob: frozenset = frozenset()        # global changes applied in imagination


def _add(pos, d):
    return pos[0] + DIRS[d][0], pos[1] + DIRS[d][1]


def _holding_key(holding) -> str:
    return ",".join(holding) or "nothing"


def goal_reached_goal(s, tile):
    return tile(s.pos)[0] == "goal"


def goal_facing_door(s, tile):
    return tile(_add(s.pos, s.facing))[0] in ("locked door", "open door")


def goal_door_with_key(s, tile):
    return "key" in s.holding and goal_facing_door(s, tile)


class WorldModelAgent:
    """Perceive -> update belief -> learn from the last transition -> plan in the learned model -> act."""

    def __init__(self, model: WorldModel, radius: int = 2, goal=goal_reached_goal,
                 curiosity: float = 0.0, max_nodes: int = 30_000):
        self.model, self.radius, self.goal, self.curiosity, self.max_nodes = model, radius, goal, curiosity, max_nodes
        self.new_episode()

    def new_episode(self):
        self.belief = Belief(self.radius)
        self.episode_records: list[tuple[int, Record]] = []
        self.prev = None
        self.plan_kind = None   # "goal" or "explore", for the demo

    # learning ------------------------------------------------------------
    def _learn(self, obs: Observation) -> None:
        b = self.belief
        if self.prev is None:
            b.update(obs)
            return
        action, pos0, facing0, holding0, here0, target0 = self.prev
        kind = "move" if action in DIRS else action
        tpos = _add(pos0, action if kind == "move" else facing0)
        changes = b.update(obs)
        moved = obs.pos == tpos if kind == "move" else obs.pos != pos0
        here1, target1 = b.tiles.get(pos0), b.tiles.get(tpos)
        gained = Counter(obs.holding) - Counter(holding0)
        lost = Counter(holding0) - Counter(obs.holding)
        local = (moved, here1 if here1 != here0 else None, target1 if target1 != target0 else None,
                 tuple(sorted(gained.elements())), tuple(sorted(lost.elements())))
        rec = Record(kind, (here0, target0, _holding_key(holding0)), local, set(), b.visible_tiles())
        # Dead reckoning: if what we saw matches the prediction, apply its predicted global
        # changes to the parts of the map that are out of view (a gate opening behind us).
        predicted, _, _ = self.model.predict(kind, rec.feats)
        if predicted is not None and predicted.local == local and predicted.glob:
            in_view = set(b.window(b.pos))
            for cell, t in b.tiles.items():
                for a, z in predicted.glob:
                    if t == a and cell not in in_view:
                        b.tiles[cell] = z
        self.episode_records.append((b.t, rec))
        self.model.add(rec)
        for cell, old, new, last_seen in changes:
            if cell in (pos0, tpos):
                continue
            if last_seen >= b.t - 1:
                rec.glob.add((old, new))  # watched it happen
            else:
                self._credit_back((old, new), last_seen)

    def _credit_back(self, change, since: int) -> None:
        """A change noticed late: credit the most recent action since then that seemed to do nothing."""
        for t, rec in reversed(self.episode_records):
            if t <= since:
                break
            if rec.kind != "move" and not any(rec.local) and not rec.glob:
                rec.glob.add(change)
                self.model.mark_dirty()
                return

    # imagination ---------------------------------------------------------
    def _tile_fn(self, s: SimState):
        tiles, over = self.belief.tiles, dict(s.overrides)

        def tile(p):
            if p in over:
                return over[p], False
            t = tiles.get(p)
            if t is None or t == PLAIN:
                return "floor", True        # optimism about the unexplored
            for a, z in s.glob:
                if t == a:
                    t = z
            return t, False
        return tile

    def _successor(self, s: SimState, action: str, tile):
        kind = "move" if action in DIRS else action
        facing = action if kind == "move" else s.facing
        tpos = _add(s.pos, facing)
        here, _ = tile(s.pos)
        target, target_unsure = tile(tpos)
        eff, unknown, novel = self.model.predict(kind, (here, target, _holding_key(s.holding)))
        if eff is None or (unknown and kind == "move" and not eff.moved):
            eff = Effect(moved=kind == "move")  # optimism: untried moves are assumed to work
        over = dict(s.overrides)
        if eff.here_to:
            over[s.pos] = eff.here_to
        if eff.target_to:
            over[tpos] = eff.target_to
        holding = list(s.holding) + list(eff.gain)
        for item in eff.lose:
            if item in holding:
                holding.remove(item)
        nxt = SimState(tpos if eff.moved else s.pos, facing, tuple(sorted(holding)),
                       frozenset(over.items()), s.glob | eff.glob)
        cost = 1.0 + (UNCERTAIN_STEP_COST if eff.moved and target_unsure else 0.0)
        return nxt, cost, novel

    def _search(self, score_fn=None):
        """Dijkstra over imagined states. With no score_fn, return the cheapest plan to the goal.
        With score_fn, return the plan to the state with the best score / (cost + 1)."""
        b = self.belief
        r0, r1, c0, c1 = b.bounds()
        start = SimState(b.pos, b.facing, b.holding)
        tie = itertools.count()
        frontier = [(0.0, next(tie), start, [])]
        best_cost = {start: 0.0}
        best = (0.0, None)
        expanded = 0
        while frontier and expanded < self.max_nodes:
            cost, _, s, path = heapq.heappop(frontier)
            if cost > best_cost.get(s, float("inf")):
                continue
            expanded += 1
            tile = self._tile_fn(s)
            if score_fn is None:
                if path and self.goal(s, tile):
                    return path
            else:
                value = score_fn(s, tile) / (cost + 1)
                if value > best[0] + 1e-9:
                    best = (value, path)
            for action in ACTIONS:
                nxt, step_cost, novel = self._successor(s, action, tile)
                if not (r0 <= nxt.pos[0] <= r1 and c0 <= nxt.pos[1] <= c1):
                    continue
                if score_fn is not None and novel and self.curiosity:
                    value = self.curiosity / (cost + step_cost + 1)
                    if value > best[0] + 1e-9:
                        best = (value, path + [action])
                c = cost + step_cost
                if c < best_cost.get(nxt, float("inf")):
                    best_cost[nxt] = c
                    heapq.heappush(frontier, (c, next(tie), nxt, path + [action]))
        return None if score_fn is None else best[1]

    def _info_gain(self, s: SimState, tile) -> float:
        tiles = self.belief.tiles
        unseen = sum(c not in tiles for c in self.belief.window(s.pos))
        unidentified = sum(tiles.get(_add(s.pos, d)) in (None, PLAIN) for d in DIRS)
        return unseen + UNIDENTIFIED_WEIGHT * unidentified

    def act(self, observation: str) -> str:
        self.perceive(observation)
        action = self.choose()
        self.commit(action)
        return action

    def perceive(self, observation: str) -> None:
        """Update the belief and learn from the transition that led to this observation."""
        self._learn(parse_observation(observation))

    def commit(self, action: str) -> None:
        """Remember the action about to be taken, so the next observation can be learned from."""
        b = self.belief
        tpos = _add(b.pos, action if action in DIRS else b.facing)
        self.prev = (action, b.pos, b.facing, b.holding, b.tiles.get(b.pos), b.tiles.get(tpos))

    def choose(self) -> str:
        """Plan to the goal in the learned model; if it isn't reachable yet, explore."""
        path = self._search() if self.goal else None
        self.plan_kind = "goal"
        if not path:
            path, self.plan_kind = self._search(self._info_gain), "explore"
        if not path:  # nothing left to see: try anything untried, else bump around
            saved, self.curiosity = self.curiosity, max(self.curiosity, 1.0)
            path, self.plan_kind = self._search(lambda s, t: 0.0), "curious"
            self.curiosity = saved
        return path[0] if path else ACTIONS[self.belief.t % len(ACTIONS)]


# --------------------------------------------------------------------------- training & evaluation
@lru_cache(maxsize=None)
def _pretrained(steps: int, radius: int) -> WorldModel:
    model = WorldModel()
    agent = WorldModelAgent(model, radius, goal=None, curiosity=2.0)
    for layout in TRAINING_LAYOUTS:
        env = GridWorld(layout, max_steps=steps, view_radius=radius)
        agent.new_episode()
        obs = env.observe()
        while not env.done:
            obs = env.step(agent.act(obs)).observation
    model.fit()
    return model


def pretrain(steps: int = 150, radius: int = 2) -> WorldModel:
    """Curiosity-driven exploration of the training maps. Deterministic, so the result is cached."""
    return copy.deepcopy(_pretrained(steps, radius))


def one_step_accuracy(model: WorldModel, layouts, steps: int = 200, seed: int = 0, radius: int = 2) -> tuple[int, int]:
    """Random-action rollouts on held-out layouts: how often does the frozen model predict the
    observed local effect (moved / tile changes / items) exactly?"""
    import random
    rng = random.Random(seed)
    hits = total = 0
    for layout, start in layouts:
        env = GridWorld(layout, max_steps=steps, view_radius=radius, start=start)
        belief = Belief(radius)
        belief.update(parse_observation(env.observe()))
        while not env.done:
            action = rng.choice(ACTIONS)
            kind = "move" if action in DIRS else action
            tpos = _add(belief.pos, action if kind == "move" else belief.facing)
            here0, target0, holding0 = belief.tiles.get(belief.pos), belief.tiles.get(tpos), belief.holding
            pred, _, _ = model.predict(kind, (here0, target0, _holding_key(holding0)))
            pos0 = belief.pos
            obs = parse_observation(env.step(action).observation)
            belief.update(obs)
            moved = obs.pos == tpos if kind == "move" else obs.pos != pos0
            h1, t1 = belief.tiles.get(pos0), belief.tiles.get(tpos)
            local = (moved, h1 if h1 != here0 else None, t1 if t1 != target0 else None,
                     tuple(sorted((Counter(obs.holding) - Counter(holding0)).elements())),
                     tuple(sorted((Counter(holding0) - Counter(obs.holding)).elements())))
            hits += pred is not None and pred.local == local
            total += 1
    return hits, total


def main() -> None:
    from .tasks import TASKS, TASKS_BY_NAME
    from .planners import WorldModelPlanner

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=TASKS_BY_NAME, help="watch the agent solve a task")
    ap.add_argument("--pretrain-steps", type=int, default=150)
    args = ap.parse_args()
    model = pretrain(args.pretrain_steps)
    if not args.task:
        print(f"Pretrained on {len(TRAINING_LAYOUTS)} training maps, {len(model)} transitions.\n")
        print("Learned rules:\n" + model.describe())
        hits, total = one_step_accuracy(model, [(t.layout, t.start) for t in TASKS])
        print(f"\nOne-step prediction accuracy on the (unseen) task maps, random actions: "
              f"{hits}/{total} = {hits / total:.1%}")
        return
    task = TASKS_BY_NAME[args.task]
    planner = WorldModelPlanner(pretrain_steps=args.pretrain_steps)
    env = task.make_env()
    planner.reset(task, env, task.seed)
    print(f"Task: {task.description} (optimal {task.optimal_steps}, limit {task.max_steps})\n")
    print("True map:\n" + env.render())
    while not env.done and not task.success(env):
        action = planner.act(env.observe(), None)
        agent = planner.agent
        print(f"\nBelief before step {env.steps + 1} ({agent.plan_kind}):\n{agent.belief.render()}\n> {action}")
        print("  " + env.step(action).message)
    print(f"\n{'Success' if task.success(env) else 'Failed'} in {env.steps} steps.")


if __name__ == "__main__":
    main()
