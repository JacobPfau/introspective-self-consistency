import unittest

from src.pipelines.sequence_completions import generate_sequence_completion_prompt


class TestSequenceCompletions(unittest.TestCase):
    def test_generate_sequence_completion_prompt(self):
        sequence = "1,2,3"
        fn_item = {"fn": f"lambda x: ({1} * x) + {1}", "offset": 0}
        prompt = generate_sequence_completion_prompt(sequence, fn_item)
        self.assertIsInstance(prompt, dict)
        self.assertIn("prompt_turns", prompt)
        self.assertIn("answer", prompt)
        self.assertIsInstance(prompt["prompt_turns"], list)
        self.assertIsInstance(prompt["answer"], str)
        self.assertGreater(len(prompt["prompt_turns"]), 0)
