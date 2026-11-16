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
