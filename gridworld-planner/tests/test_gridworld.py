import unittest

from gridworld_planner import GridWorld, plan

# A corridor exercising every mechanic in order: key, door, switch, gate, goal.
CORRIDOR = "########\n#@kDsXG#\n########"


def run(env, actions):
    return [env.step(a) for a in actions][-1]


class MovementTests(unittest.TestCase):
    def test_walls_block_and_set_facing(self):
        env = GridWorld(CORRIDOR)
        res = env.step("up")
        self.assertEqual(env.agent, (1, 1))
        self.assertEqual(env.facing, "up")
        self.assertIn("bump into the wall", res.message)

    def test_move_onto_floor(self):
        env = GridWorld(CORRIDOR)
        env.step("right")
        self.assertEqual(env.agent, (1, 2))
        self.assertEqual(env.facing, "right")

    def test_invalid_action_raises(self):
        with self.assertRaises(ValueError):
            GridWorld(CORRIDOR).step("jump")

    def test_bad_layouts_rejected(self):
        with self.assertRaises(ValueError):
            GridWorld("#.#")  # no agent
        with self.assertRaises(ValueError):
            GridWorld("#@?#")  # unknown character


class ItemTests(unittest.TestCase):
    def test_pick_up_key(self):
        env = GridWorld(CORRIDOR)
        res = run(env, ["right", "pick_up"])
        self.assertEqual(env.holding, ["key"])
        self.assertIn("Holding: key.", res.observation)
        self.assertEqual(env.cell((1, 2)), ".")

    def test_pick_up_nothing(self):
        env = GridWorld(CORRIDOR)
        res = env.step("pick_up")
        self.assertIn("nothing here to pick up", res.message)
        self.assertEqual(env.holding, [])

    def test_locked_door_needs_key(self):
        env = GridWorld("#####\n#@Dk#\n#####")
        env.step("right")
        self.assertEqual(env.agent, (1, 1))
        res = env.step("use")
        self.assertIn("need a key", res.message)
        self.assertEqual(env.cell((1, 2)), "D")

    def test_key_unlocks_door_and_is_consumed(self):
        env = GridWorld(CORRIDOR)
        res = run(env, ["right", "pick_up", "use"])
        self.assertIn("unlock the door", res.message)
        self.assertEqual(env.holding, [])
        self.assertEqual(env.cell((1, 3)), "d")
        env.step("right")
        self.assertEqual(env.agent, (1, 3))

    def test_gate_closed_until_switch_pressed(self):
        env = GridWorld(CORRIDOR)
        run(env, ["right", "pick_up", "use", "right", "right"])
        self.assertEqual(env.agent, (1, 4))  # standing on the switch
        res = env.step("right")
        self.assertEqual(env.agent, (1, 4))
        self.assertIn("closed gate", res.message)
        res = env.step("use")
        self.assertTrue(env.switch_on)
        self.assertIn("press the switch", res.message)
        self.assertIn("already on", env.step("use").message)
        env.step("right")
        self.assertEqual(env.agent, (1, 5))

    def test_using_gate_directly_does_nothing(self):
        env = GridWorld("#####\n#@X.#\n#####")
        env.step("right")
        self.assertIn("won't budge", env.step("use").message)


class EpisodeTests(unittest.TestCase):
    SOLUTION = ["right", "pick_up", "use", "right", "right", "use", "right", "right"]

    def test_reaching_goal_ends_episode(self):
        env = GridWorld(CORRIDOR)
        res = run(env, self.SOLUTION)
        self.assertTrue(res.done)
        self.assertEqual(res.reward, 1.0)
        self.assertTrue(env.reached_goal)
        after = env.step("left")
        self.assertTrue(after.done)
        self.assertEqual(env.agent, (1, 6))

    def test_step_cost_and_step_limit(self):
        env = GridWorld(CORRIDOR, max_steps=2)
        self.assertLess(env.step("up").reward, 0)
        res = env.step("up")
        self.assertTrue(res.done)
        self.assertFalse(env.reached_goal)
        self.assertIn("out of steps", res.message)

    def test_reset_restores_initial_state(self):
        env = GridWorld(CORRIDOR)
        run(env, self.SOLUTION)
        env.reset()
        self.assertEqual((env.agent, env.holding, env.switch_on, env.steps), ((1, 1), [], False, 0))
        self.assertEqual(env.cell((1, 2)), "k")
        self.assertEqual(env.open_doors, set())


class ObservationTests(unittest.TestCase):
    def test_observation_describes_surroundings(self):
        obs = GridWorld(CORRIDOR).observe()
        self.assertIn("row 1, column 1", obs)
        self.assertIn("Holding: nothing.", obs)
        self.assertIn("up: wall", obs)
        self.assertIn("right: key", obs)
        self.assertIn("key (1 right)", obs)
        self.assertIn("locked door (2 right)", obs)
        self.assertNotIn("switch", obs)  # 3 cells away, beyond view radius 2

    def test_observation_reflects_opened_things(self):
        env = GridWorld(CORRIDOR)
        res = run(env, ["right", "pick_up", "use", "right", "right", "use"])
        self.assertIn("open door (1 left)", res.observation)
        self.assertIn("open gate (1 right)", res.observation)
        self.assertIn("Here: switch.", res.observation)

    def test_render(self):
        env = GridWorld(CORRIDOR)
        self.assertEqual(env.render().splitlines()[1], "#@kDsXG#")


class PlannerTests(unittest.TestCase):
    def test_plans_default_layout(self):
        env = GridWorld()
        actions = plan(env)
        self.assertIsNotNone(actions)
        self.assertEqual(env.steps, 0)  # planning does not touch the env
        res = run(env, actions)
        self.assertTrue(res.done and env.reached_goal)

    def test_plan_is_shortest_on_corridor(self):
        self.assertEqual(plan(GridWorld(CORRIDOR)), EpisodeTests.SOLUTION)

    def test_unsolvable_returns_none(self):
        self.assertIsNone(plan(GridWorld("#####\n#@DG#\n#####")))


if __name__ == "__main__":
    unittest.main()
