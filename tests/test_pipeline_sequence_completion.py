import unittest
from unittest.mock import patch

from src.pipelines.sequence_completions import generate_sequence_completion_prompt


class TestSequenceCompletions(unittest.TestCase):
    @patch("src.prompt_generation.prompt_loader.get_original_cwd")
    def test_generate_sequence_completion_prompt(self, cwd_patch):

        cwd_patch.return_value = ""

        sequence = "1,2,3"
        fn_item = {"fn": f"lambda x: ({1} * x) + {1}", "offset": 0}
        prompt = generate_sequence_completion_prompt(sequence, fn_item)
        self.assertIsInstance(prompt, dict)
        self.assertIn("prompt_turns", prompt)
        self.assertIn("answer", prompt)
        self.assertIsInstance(prompt["prompt_turns"], list)
        self.assertIsInstance(prompt["answer"], str)
        self.assertGreater(len(prompt["prompt_turns"]), 0)
