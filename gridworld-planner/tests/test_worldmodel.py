import unittest

from gridworld_planner import worldmodel as wm
from gridworld_planner.env import GridWorld
from gridworld_planner.planners import WorldModelPlanner
from gridworld_planner.runner import run_episode
from gridworld_planner.tasks import TASKS, TASKS_BY_NAME


class PerceptionTests(unittest.TestCase):
    def test_parse_observation_matches_env_state(self):
        env = GridWorld()
        env.step("down")
        env.step("right")
        env.step("pick_up")
        obs = wm.parse_observation(env.observe("hello"))
        self.assertEqual(obs.pos, env.agent)
        self.assertEqual(obs.facing, "right")
        self.assertEqual(obs.holding, ("key",))
        self.assertEqual(obs.here, "floor")
        self.assertEqual(obs.around["right"], "floor")
        self.assertIn(("locked door", (0, 2)), obs.nearby)
        self.assertEqual(obs.message, "hello")

    def test_belief_identifies_neighbours_and_marks_the_rest_plain(self):
        env = GridWorld()
        belief = wm.Belief(radius=2)
        belief.update(wm.parse_observation(env.observe()))
        r, c = env.agent
        self.assertEqual(belief.tiles[(r - 1, c)], "wall")
        self.assertEqual(belief.tiles[(r, c + 1)], "floor")
        self.assertEqual(belief.tiles[(r + 1, c + 1)], "key")
        self.assertEqual(belief.tiles[(r + 2, c + 2)], wm.PLAIN)
        self.assertNotIn((r + 3, c), belief.tiles)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = wm.pretrain()

    def predict(self, kind, here, target, holding="nothing"):
        effect, unknown, _ = self.model.predict(kind, (here, target, holding))
        self.assertFalse(unknown, f"{kind} {here} {target} {holding} fell outside the learned tree")
        return effect

    def test_learns_the_rules_from_the_training_maps(self):
        self.assertTrue(self.predict("move", "floor", "floor").moved)
        self.assertFalse(self.predict("move", "floor", "wall").moved)
        self.assertFalse(self.predict("move", "key", "locked door").moved)   # generalises: here is irrelevant
        self.assertFalse(self.predict("move", "floor", "closed gate").moved)
        self.assertEqual(self.predict("pick_up", "key", "wall"), wm.Effect(here_to="floor", gain=("key",)))
        self.assertEqual(self.predict("pick_up", "floor", "key"), wm.Effect())
        self.assertEqual(self.predict("use", "floor", "locked door", "key"),
                         wm.Effect(target_to="open door", lose=("key",)))
        self.assertEqual(self.predict("use", "floor", "locked door"), wm.Effect())
        self.assertEqual(self.predict("use", "switch", "key").glob, {("closed gate", "open gate")})

    def test_perfect_one_step_predictions_on_unseen_task_maps(self):
        hits, total = wm.one_step_accuracy(self.model, [(t.layout, t.start) for t in TASKS])
        self.assertEqual(hits, total)

    def test_unseen_global_change_cannot_refute_a_hypothesis(self):
        opens = wm.Effect(glob=frozenset({("closed gate", "open gate")}))
        nothing_seen = wm.Record("use", ("switch", "wall", "nothing"), wm.Effect().local, set(), frozenset())
        gate_in_view = wm.Record("use", ("switch", "wall", "nothing"), wm.Effect().local, set(),
                                 frozenset({"closed gate"}))
        self.assertTrue(wm.consistent(opens, nothing_seen))
        self.assertFalse(wm.consistent(opens, gate_in_view))


class CreditAssignmentTests(unittest.TestCase):
    def test_gate_seen_open_later_is_credited_to_the_switch_press(self):
        # The gate is 4 squares from the switch, out of view when the switch is pressed.
        env = GridWorld("#########\n#s.@.X..#\n#########")
        agent = wm.WorldModelAgent(wm.WorldModel(), goal=None)
        obs = env.observe()
        for action in ["left", "left", "use", "right", "right"]:
            agent.perceive(obs)
            agent.commit(action)
            obs = env.step(action).observation
        agent.perceive(obs)
        effect, _, _ = agent.model.predict("use", ("switch", "floor", "nothing"))
        self.assertEqual(effect.glob, {("closed gate", "open gate")})

    def test_dead_reckoning_updates_out_of_view_cells(self):
        env = GridWorld("#########\n#s.@.X..#\n#########")
        agent = wm.WorldModelAgent(wm.pretrain(), goal=None)
        obs = env.observe()
        for action in ["left", "left", "use"]:
            agent.perceive(obs)
            agent.commit(action)
            obs = env.step(action).observation
        agent.perceive(obs)
        self.assertEqual(agent.belief.tiles[(1, 5)], "open gate")


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planner = WorldModelPlanner()

    def test_solves_every_task_without_failed_actions(self):
        for task in TASKS:
            with self.subTest(task=task.name):
                r = run_episode(task, self.planner, feedback=False)
                self.assertTrue(r.success)
                self.assertEqual(r.failed_actions, 0)

    def test_optimal_when_the_whole_chain_is_in_view_early(self):
        for name in ("walk_to_door", "key_then_door", "unlock_and_pass"):
            task = TASKS_BY_NAME[name]
            self.assertEqual(run_episode(task, self.planner, feedback=True).steps_used, task.optimal_steps)

    def test_episodes_start_from_the_same_pretrained_model(self):
        task = TASKS_BY_NAME["two_room_chain"]
        first = run_episode(task, self.planner, feedback=True, episode=0)
        second = run_episode(task, self.planner, feedback=True, episode=1)
        self.assertEqual(first.steps_used, second.steps_used)
        self.assertEqual(len(self.planner.base), len(wm.pretrain()))


if __name__ == "__main__":
    unittest.main()
