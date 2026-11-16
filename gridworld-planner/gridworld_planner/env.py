"""A small text gridworld: walls, a key, a locked door, a switch-operated gate and a goal."""
from __future__ import annotations

import random
from dataclasses import dataclass

# Layout characters. '@' marks the agent's start and is replaced by floor.
DEFAULT_LAYOUT = """
##########
#@..#....#
#.k.D..s.#
#...#....#
#####X####
#....G...#
##########
"""

DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ACTIONS = (*DIRS, "pick_up", "use")
NAMES = {
    "#": "wall", ".": "floor", "k": "key", "D": "locked door", "d": "open door",
    "s": "switch", "X": "closed gate", "x": "open gate", "G": "goal",
}
LAYOUT_CHARS = set("#.kDsXG@")
BLOCKING = set("#DX")
STEP_COST = -0.01


@dataclass
class StepResult:
    observation: str
    reward: float
    done: bool
    message: str
    failed: bool = False  # True if the action had no effect (bumped, locked, nothing to use...)


def _offset(dr: int, dc: int) -> str:
    parts = []
    if dr:
        parts.append(f"{abs(dr)} {'down' if dr > 0 else 'up'}")
    if dc:
        parts.append(f"{abs(dc)} {'right' if dc > 0 else 'left'}")
    return ", ".join(parts)


class GridWorld:
    """Positions are (row, col). Moving sets the facing direction even if blocked.

    - ``pick_up`` takes a key from the agent's own cell.
    - ``use`` presses a switch the agent stands on (opening every gate, once),
      otherwise acts on the cell the agent faces: a locked door is unlocked if
      the agent holds a key (the key is consumed).
    """

    def __init__(self, layout: str = DEFAULT_LAYOUT, max_steps: int = 200, view_radius: int = 2,
                 seed: int | None = None, start: tuple[int, int] | None = None):
        """``start`` overrides the '@' position, e.g. to start the agent on a switch.

        The dynamics are deterministic; ``seed`` seeds ``self.rng`` on every reset so
        any stochastic extension stays reproducible.
        """
        self.layout = layout
        self.max_steps = max_steps
        self.view_radius = view_radius
        self.seed = seed
        self.start = start
        self.reset()

    def reset(self) -> str:
        self.rng = random.Random(self.seed)
        rows = self.layout.strip("\n").splitlines()
        width = max(len(r) for r in rows)
        self.grid = [list(r.ljust(width, "#")) for r in rows]
        self.agent = None
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch not in LAYOUT_CHARS:
                    raise ValueError(f"unknown layout character {ch!r} at ({r}, {c})")
                if ch == "@":
                    if self.agent is not None:
                        raise ValueError("layout has more than one agent '@'")
                    self.agent, row[c] = (r, c), "."
        if self.start is not None:
            r, c = self.start
            if not (0 <= r < len(self.grid) and 0 <= c < width) or self.grid[r][c] in BLOCKING:
                raise ValueError(f"start {self.start} is not walkable")
            self.agent = tuple(self.start)
        if self.agent is None:
            raise ValueError("layout needs an agent '@' or a start position")
        self.facing = "up"
        self.holding: list[str] = []
        self.open_doors: set[tuple[int, int]] = set()
        self.switch_on = False
        self.steps = 0
        self.done = False
        self.reached_goal = False
        return self.observe("You wake up in a gridworld.")

    # --- queries -----------------------------------------------------------
    def cell(self, pos: tuple[int, int]) -> str:
        """Effective tile at pos, accounting for opened doors and gates."""
        r, c = pos
        if not (0 <= r < len(self.grid) and 0 <= c < len(self.grid[r])):
            return "#"
        ch = self.grid[r][c]
        if ch == "D" and pos in self.open_doors:
            return "d"
        if ch == "X" and self.switch_on:
            return "x"
        return ch

    def _ahead(self) -> tuple[int, int]:
        dr, dc = DIRS[self.facing]
        return self.agent[0] + dr, self.agent[1] + dc

    def state_key(self) -> tuple:
        """Hashable summary of everything that affects future dynamics."""
        keys = tuple((r, c) for r, row in enumerate(self.grid) for c, ch in enumerate(row) if ch == "k")
        return (self.agent, self.facing, tuple(self.holding),
                frozenset(self.open_doors), self.switch_on, keys)

    # --- dynamics ----------------------------------------------------------
    def step(self, action: str) -> StepResult:
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}; expected one of {ACTIONS}")
        if self.done:
            msg = "The episode is over. Call reset() to play again."
            return StepResult(self.observe(msg), 0.0, True, msg, failed=True)
        self.steps += 1
        if action in DIRS:
            msg, failed = self._move(action)
        elif action == "pick_up":
            msg, failed = self._pick_up()
        else:
            msg, failed = self._use()
        reward = STEP_COST
        if self.cell(self.agent) == "G":
            self.done = self.reached_goal = True
            reward = 1.0
            msg += " You reached the goal!"
        elif self.steps >= self.max_steps:
            self.done = True
            msg += " You are out of steps."
        return StepResult(self.observe(msg), reward, self.done, msg, failed)

    def waste_step(self, message: str) -> StepResult:
        """Consume a step without acting (e.g. the planner produced no valid action)."""
        if not self.done:
            self.steps += 1
            if self.steps >= self.max_steps:
                self.done = True
        return StepResult(self.observe(message), STEP_COST, self.done, message, failed=True)

    # Handlers return (message, failed).
    def _move(self, direction: str) -> tuple[str, bool]:
        self.facing = direction
        target = self._ahead()
        tile = self.cell(target)
        if tile in BLOCKING:
            return f"You bump into the {NAMES[tile]}.", True
        self.agent = target
        return f"You move {direction}.", False

    def _pick_up(self) -> tuple[str, bool]:
        r, c = self.agent
        if self.grid[r][c] == "k":
            self.grid[r][c] = "."
            self.holding.append("key")
            return "You pick up the key.", False
        return "There is nothing here to pick up.", True

    def _use(self) -> tuple[str, bool]:
        if self.cell(self.agent) == "s":
            if self.switch_on:
                return "The switch is already on.", True
            self.switch_on = True
            return "You press the switch. You hear a gate grind open.", False
        target = self._ahead()
        tile = self.cell(target)
        if tile == "D":
            if "key" in self.holding:
                self.holding.remove("key")
                self.open_doors.add(target)
                return "You unlock the door with the key.", False
            return "The door is locked. You need a key.", True
        if tile == "X":
            return "The gate won't budge. Perhaps a switch controls it.", True
        return "There is nothing to use here.", True

    # --- text output -------------------------------------------------------
    def observe(self, message: str = "") -> str:
        r, c = self.agent
        lines = [message] if message else []
        lines.append(f"Step {self.steps}. You are at row {r}, column {c}, facing {self.facing}.")
        lines.append("Holding: " + (", ".join(self.holding) or "nothing") + ".")
        around = ", ".join(f"{d}: {NAMES[self.cell((r + dr, c + dc))]}" for d, (dr, dc) in DIRS.items())
        lines.append(f"Here: {NAMES[self.cell(self.agent)]}. Around you: {around}.")
        seen = []
        rad = self.view_radius
        for rr in range(r - rad, r + rad + 1):
            for cc in range(c - rad, c + rad + 1):
                tile = self.cell((rr, cc))
                if tile not in "#." and (rr, cc) != self.agent:
                    seen.append(f"{NAMES[tile]} ({_offset(rr - r, cc - c)})")
        lines.append("Nearby: " + ("; ".join(seen) or "nothing of interest") + ".")
        return "\n".join(lines)

    def render(self) -> str:
        """ASCII map of the current state, with '@' for the agent."""
        out = []
        for r, row in enumerate(self.grid):
            out.append("".join("@" if (r, c) == self.agent else self.cell((r, c)) for c in range(len(row))))
        return "\n".join(out)
