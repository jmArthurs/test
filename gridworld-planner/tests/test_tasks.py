import unittest

from gridworld_planner import plan
from gridworld_planner.tasks import TASKS, TASKS_BY_NAME

# Hand-coded optimal (open-loop) policies, one per task.
POLICIES = {
    "walk_to_door": ["right"],
    "key_then_door": ["right", "pick_up", "right"],
    "unlock_and_pass": ["right", "pick_up", "use", "right", "right"],
    "switch_key_door": ["use", "right", "pick_up", "use", "right", "right", "right"],
    "two_room_chain": ["left", "use", "down", "pick_up", "right", "use",
                       "right", "right", "right", "right"],
}


class TaskTests(unittest.TestCase):
    def test_ladder_of_step_counts(self):
        self.assertEqual([t.optimal_steps for t in TASKS], [1, 3, 5, 7, 10])
        self.assertEqual(set(POLICIES), set(TASKS_BY_NAME))

    def test_hand_coded_policies_solve_each_task(self):
        for task in TASKS:
            with self.subTest(task=task.name):
                env = task.make_env()
                actions = POLICIES[task.name]
                self.assertEqual(len(actions), task.optimal_steps)
                self.assertLessEqual(len(actions), task.max_steps)
                for action in actions:
                    self.assertFalse(task.success(env), "succeeded before the policy finished")
                    self.assertFalse(env.step(action).done and not env.reached_goal)
                self.assertTrue(task.success(env))
                self.assertEqual(env.steps, task.optimal_steps)

    def test_policies_are_optimal(self):
        # BFS finds the shortest plan, so matching its length proves no shorter solution exists.
        for task in TASKS:
            with self.subTest(task=task.name):
                shortest = plan(task.make_env(), task.success)
                self.assertEqual(len(shortest), task.optimal_steps)

    def test_step_limit_enforced(self):
        task = TASKS_BY_NAME["two_room_chain"]
        env = task.make_env()
        for _ in range(task.max_steps):
            res = env.step("up")
        self.assertTrue(res.done)
        self.assertFalse(task.success(env))

    def test_seed_is_fixed_and_reproducible(self):
        for task in TASKS:
            a, b = task.make_env(), task.make_env()
            self.assertEqual(a.seed, task.seed)
            self.assertEqual(a.rng.random(), b.rng.random())
            self.assertEqual(a.render(), b.render())

    def test_start_override(self):
        env = TASKS_BY_NAME["switch_key_door"].make_env()
        self.assertEqual(env.agent, (1, 1))
        self.assertEqual(env.cell(env.agent), "s")


if __name__ == "__main__":
    unittest.main()
