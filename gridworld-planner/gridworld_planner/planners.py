"""Planner interface plus three planners: random, oracle (full map) and LLM (text only).

The runner calls ``reset`` once per episode, then ``act`` once per step with the current
text observation and, if failure feedback is on and the last action failed, a short
description of what went wrong. ``act`` returns an action string; anything that is not
in ACTIONS is treated by the runner as an invalid reply that wastes a step.
"""
from __future__ import annotations

import random
import re

from .env import ACTIONS, GridWorld
from .planner import plan
from .tasks import Task


class Planner:
    name = "planner"

    def reset(self, task: Task, env: GridWorld, seed: int) -> None:
        pass

    def act(self, observation: str, feedback: str | None) -> str:
        raise NotImplementedError

    @property
    def model(self) -> str:
        """Model(s) that served this episode; empty for non-LLM planners."""
        return ""


class RandomPlanner(Planner):
    """Uniformly random actions, seeded per episode."""
    name = "random"

    def reset(self, task, env, seed):
        self.rng = random.Random(seed)

    def act(self, observation, feedback):
        return self.rng.choice(ACTIONS)


class OraclePlanner(Planner):
    """Reads the full map and state from the env and follows a BFS-optimal plan."""
    name = "oracle"

    def reset(self, task, env, seed):
        self.task, self.env, self.queue = task, env, []

    def act(self, observation, feedback):
        if not self.queue:  # (re)plan from the true current state
            self.queue = plan(self.env, self.task.success) or ["up"]
        return self.queue.pop(0)


SYSTEM_PROMPT = """You control an agent in a text gridworld. You cannot see the whole map: each turn \
you get a description of the agent's surroundings, and you must choose one action.

Actions: up, down, left, right, pick_up, use.
- Moving also turns the agent to face that direction, even when the move is blocked.
- Walls, locked doors and closed gates block movement. Open doors and open gates do not.
- pick_up picks up a key on the agent's own square.
- use presses a switch the agent is standing on; otherwise it unlocks the locked door the \
agent is facing, if the agent holds a key (the key is used up).
- Pressing a switch opens the gates. Closed gates cannot be opened with a key.
- Rows count downward and columns count rightward; "1 down, 2 right" is relative to the agent.

Every action, including a failed one, uses up one step of a fixed step budget. Think briefly, \
then end your reply with exactly one line of the form:
ACTION: <action>"""

_ACTION_RE = re.compile(r"ACTION:\s*\**\s*([A-Za-z_ ]+)", re.IGNORECASE)
_ALIASES = {"pickup": "pick_up", "pick up": "pick_up", "pick": "pick_up"}


def parse_action(text: str) -> str | None:
    """Pull the action out of a reply; None if there is no valid one."""
    matches = _ACTION_RE.findall(text or "")
    if matches:
        raw = matches[-1]
    else:  # no ACTION line: accept a bare action on the last line
        lines = (text or "").strip().splitlines()
        raw = lines[-1] if lines else ""
    words = raw.strip().strip(".*`'\"").lower().split()
    for n in (2, 1):  # "pick up the key" -> "pick up"; "right." -> "right"
        word = " ".join(words[:n])
        word = _ALIASES.get(word, word.replace(" ", "_"))
        if word in ACTIONS:
            return word
    return None


class LLMPlanner(Planner):
    """Asks an LLM backend for each action, keeping the whole episode as one chat."""
    name = "llm"

    def __init__(self, backend):
        self.backend = backend

    def reset(self, task, env, seed):
        self.seed = seed
        self.messages: list[dict] = []
        self.models: list[str] = []
        self.goal = (f"Task: {task.description}\n"
                     f"You have at most {task.max_steps} steps. The episode ends as soon as the task "
                     f"is complete.")

    def user_message(self, observation: str, feedback: str | None) -> str:
        parts = [self.goal] if not self.messages else []
        if feedback:
            parts.append(f"Feedback: {feedback}")
        parts.append(f"Observation:\n{observation}")
        return "\n\n".join(parts)

    def act(self, observation, feedback):
        self.messages.append({"role": "user", "content": self.user_message(observation, feedback)})
        reply = self.backend.respond(SYSTEM_PROMPT, self.messages, seed=self.seed)
        self.messages.append(reply.message)
        if reply.model not in self.models:
            self.models.append(reply.model)
        return parse_action(reply.text) or f"<invalid: {reply.text[-80:]!r}>"

    @property
    def model(self):
        return "|".join(self.models)
