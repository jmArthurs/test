import contextlib
import csv
import io
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from gridworld_planner import evaluate
from gridworld_planner.llm import ApiBackend, ApiSettings, LocalBackend, LocalSettings, Reply, load_config
from gridworld_planner.planners import LLMPlanner, OraclePlanner, RandomPlanner, parse_action
from gridworld_planner.plot import summarize, wilson
from gridworld_planner.runner import run_episode
from gridworld_planner.tasks import TASKS, TASKS_BY_NAME


class ScriptedBackend:
    """Fake LLM backend: replays canned replies and records what it was sent."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def respond(self, system, messages, seed=None):
        self.calls.append([dict(m) for m in messages])
        text = self.replies.pop(0)
        return Reply(text, {"role": "assistant", "content": text}, "fake-model")


class ParseActionTests(unittest.TestCase):
    def test_parses_action_lines(self):
        cases = {
            "I should go right.\nACTION: right": "right",
            "ACTION: up\nwait, no\nACTION: down": "down",
            "action: **pick up** the key": "pick_up",
            "ACTION: Use.": "use",
            "left": "left",
            "ACTION: jump": None,
            "": None,
        }
        for text, want in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_action(text), want)


class SimplePlannerTests(unittest.TestCase):
    def test_oracle_solves_every_task_optimally(self):
        for task in TASKS:
            for fb in (False, True):
                r = run_episode(task, OraclePlanner(), fb)
                self.assertTrue(r.success, task.name)
                self.assertEqual(r.steps_used, task.optimal_steps)
                self.assertEqual(r.failed_actions, 0)

    def test_random_is_reproducible_per_seed(self):
        task = TASKS_BY_NAME["two_room_chain"]
        a = run_episode(task, RandomPlanner(), False, episode=3)
        b = run_episode(task, RandomPlanner(), False, episode=3)
        self.assertEqual(a, b)
        self.assertEqual(a.seed, task.seed * 1000 + 3)


class LLMPlannerTests(unittest.TestCase):
    TASK = TASKS_BY_NAME["unlock_and_pass"]  # optimal: right pick_up use right right

    def run_llm(self, replies, feedback):
        backend = ScriptedBackend(replies)
        return run_episode(self.TASK, LLMPlanner(backend), feedback), backend

    def test_solves_task_from_replies(self):
        replies = [f"thinking...\nACTION: {a}" for a in ["right", "pick_up", "use", "right", "right"]]
        r, backend = self.run_llm(replies, feedback=True)
        self.assertTrue(r.success)
        self.assertEqual(r.steps_used, 5)
        self.assertEqual(r.model, "fake-model")
        first = backend.calls[0][0]["content"]
        self.assertIn("Task: " + self.TASK.description, first)
        self.assertIn("Observation:", first)
        self.assertNotIn("Task:", backend.calls[1][-1]["content"])  # goal only stated once
        self.assertEqual([m["role"] for m in backend.calls[1]], ["user", "assistant", "user"])

    def test_failure_feedback_only_when_enabled(self):
        replies = ["ACTION: up", "ACTION: right", "ACTION: pick_up", "ACTION: use",
                   "ACTION: right", "ACTION: right"]
        on, backend_on = self.run_llm(replies, feedback=True)
        off, backend_off = self.run_llm(replies, feedback=False)
        msg_on = backend_on.calls[1][-1]["content"]
        self.assertIn('Feedback: Your last action "up" failed: You bump into the wall.', msg_on)
        self.assertNotIn("Feedback", backend_off.calls[1][-1]["content"])
        self.assertNotIn("bump", backend_off.calls[1][-1]["content"])
        for r in (on, off):
            self.assertTrue(r.success)
            self.assertEqual((r.steps_used, r.failed_actions), (6, 1))

    def test_invalid_reply_wastes_a_step(self):
        replies = ["I'm not sure.", "ACTION: right", "ACTION: pick_up", "ACTION: use",
                   "ACTION: right", "ACTION: right"]
        r, backend = self.run_llm(replies, feedback=True)
        self.assertTrue(r.success)
        self.assertEqual((r.steps_used, r.invalid_replies, r.failed_actions), (6, 1, 0))
        self.assertIn("did not end with a valid action", backend.calls[1][-1]["content"])

    def test_step_limit_ends_episode(self):
        r, _ = self.run_llm(["ACTION: up"] * self.TASK.max_steps, feedback=False)
        self.assertFalse(r.success)
        self.assertEqual(r.steps_used, self.TASK.max_steps)


class BackendTests(unittest.TestCase):
    def test_api_backend_request_and_refusal(self):
        block = SimpleNamespace(type="text", text="ACTION: up")
        resp = SimpleNamespace(content=[block], stop_reason="end_turn", model="claude-opus-5-5")
        client = mock.MagicMock()
        client.beta.messages.create.return_value = resp
        reply = ApiBackend(ApiSettings(effort="low"), client=client).respond("sys", [{"role": "user", "content": "hi"}])
        kw = client.beta.messages.create.call_args.kwargs
        self.assertEqual(kw["fallbacks"], "default")
        self.assertEqual(kw["output_config"], {"effort": "low"})
        self.assertEqual(kw["model"], "claude-opus-5-5")
        self.assertEqual(reply.text, "ACTION: up")
        self.assertIs(reply.message["content"], resp.content)  # echoed unchanged

        resp.stop_reason = "refusal"
        client.messages.create.return_value = resp
        reply = ApiBackend(ApiSettings(fallbacks=False), client=client).respond("sys", [])
        self.assertTrue(client.messages.create.called)
        self.assertEqual(reply.text, "")

    def test_local_backend_calls_ollama_chat(self):
        sent = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return json.dumps({"model": "m", "message": {"content": "ACTION: left"}}).encode()

        def fake_urlopen(req, timeout):
            sent["url"], sent["body"] = req.full_url, json.loads(req.data)
            return FakeResponse()

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            reply = LocalBackend(LocalSettings(model="m")).respond("sys", [{"role": "user", "content": "hi"}], seed=7)
        self.assertEqual(sent["url"], "http://localhost:11434/api/chat")
        self.assertEqual(sent["body"]["messages"][0], {"role": "system", "content": "sys"})
        self.assertEqual(sent["body"]["options"]["seed"], 7)
        self.assertEqual(reply.text, "ACTION: left")

    def test_config_switch(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.toml"
            p.write_text('[llm]\nbackend = "local"\n[llm.local]\nmodel = "qwen"\n')
            cfg = load_config(p)
            self.assertEqual((cfg.backend, cfg.model), ("local", "qwen"))
            self.assertEqual(load_config(p, backend="api").model, "claude-opus-5-5")
            p.write_text('[llm]\nbackend = "cloud"\n')
            with self.assertRaises(ValueError):
                load_config(p)
            p.write_text('[llm.api]\nmodle = "x"\n')
            with self.assertRaises(ValueError):
                load_config(p)


class EvaluateTests(unittest.TestCase):
    def test_wilson(self):
        lo, hi = wilson(0, 20)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.161, places=3)
        lo, hi = wilson(10, 20)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_end_to_end_csv_resume_and_plot(self):
        with tempfile.TemporaryDirectory() as d:
            out, png = Path(d) / "r.csv", Path(d) / "p.png"
            has_mpl = importlib.util.find_spec("matplotlib") is not None
            args = ["--episodes", "2", "--out", str(out), "--plot", str(png) if has_mpl else ""]
            with contextlib.redirect_stderr(io.StringIO()):
                evaluate.main(args)
            with out.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3 * 2 * 5 * 2)  # planners x feedback x tasks x episodes
            self.assertEqual(list(rows[0]), evaluate.COLUMNS)
            with contextlib.redirect_stderr(io.StringIO()):
                evaluate.main(args[:-2] + ["--plot", "", "--resume", "--episodes", "3"])
            with out.open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 3 * 2 * 5 * 3)
            summary = summarize(out)
            self.assertEqual(summary[("oracle", True)][10], (3, 3))
            self.assertEqual(summary[("worldmodel", False)][10], (3, 3))
            if has_mpl:
                self.assertGreater(png.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
