# gridworld-planner

A tiny text gridworld in pure Python (no dependencies) plus a breadth-first planner.

Tiles: `#` wall, `.` floor, `@` agent start, `k` key, `D` door locked with the key,
`s` switch, `X` gate opened by the switch, `G` goal.

Actions: `up`, `down`, `left`, `right`, `pick_up`, `use`.

- Moving changes the direction you face, even when blocked.
- `pick_up` takes a key from your own cell.
- `use` presses a switch you stand on; otherwise it unlocks the door you face if you hold a key (the key is used up).

`env.step(action)` returns a `StepResult(observation, reward, done, message)`. The observation is text:
your position and facing, what you hold, the four neighbouring tiles, and notable objects within 2 cells.
Reward is -0.01 per step and 1.0 for reaching the goal.

```python
from gridworld_planner import GridWorld, plan
env = GridWorld()            # or GridWorld(your_layout_string)
print(env.step("down").observation)
print(plan(env))             # shortest action list to the goal, or None
```

```
python -m gridworld_planner          # watch the planner solve the default map
python -m gridworld_planner --play   # play it yourself
python -m unittest                   # run the tests
```

## Tasks

`gridworld_planner.tasks.TASKS` is a ladder of fixed tasks. Each has a layout, a seed,
a success check and a step limit; build one with `task.make_env()`.

| name | optimal steps | max steps | success |
|---|---|---|---|
| `walk_to_door` | 1 | 5 | facing a door |
| `key_then_door` | 3 | 10 | holding the key and facing the door |
| `unlock_and_pass` | 5 | 15 | goal reached (behind the locked door) |
| `switch_key_door` | 7 | 20 | goal reached; start on the switch, gate behind the door |
| `two_room_chain` | 10 | 30 | goal reached; switch and key in room 1, gate in room 2 |

`tests/test_tasks.py` solves each one with a hand-coded policy and checks with the BFS
planner that no shorter solution exists. `plan(env, success)` accepts any success check.
