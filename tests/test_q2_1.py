import unittest

from src.evals.config import Q21LogprobInequalityConfig
from src.evals.q2_1_logprob_inequality import list_rindex


class TestQ21(unittest.TestCase):
    def test_list_rindex_with_unique_tokens(self):
        test_string = list("abcdefghijklmnopqrstuvwxyz")

        self.assertEqual(list_rindex(test_string, "z"), 25)
        self.assertEqual(list_rindex(test_string, "a"), 0)
        self.assertEqual(list_rindex(test_string, "b"), 1)

    def test_list_rindex_with_repeated_token(self):
        test_string = [
            "a",
            "a",
            "a",
            "b",
            "c",
            "b",
            "b",
            "c",
        ]

        test_tokens = ["a", "b", "c", "a", "b"]

        tkn_indices = []
        for tkn in test_tokens:
            idx = list_rindex(test_string, tkn, tkn_indices)
            tkn_indices.append(idx)

        self.assertListEqual(tkn_indices, [2, 6, 7, 1, 5])

    def test_eval_config_from_dict_initializes_correctly(self):
        test_dict = {
            "task": "q2_1_logprob_inequality",
            "model": "text-davinci-003",
            "num_shots": 13,
            "num_invalid": 1,
            "num_valid": 2,
            "few_shot_prompt_type": "random",
            "seed": 123,
        }

        config = Q21LogprobInequalityConfig.from_dict(test_dict)

        self.assertEqual(config.task, "q2_1_logprob_inequality")
        self.assertEqual(config.num_shots, 13)
        self.assertEqual(config.num_invalid, 1)
        self.assertEqual(config.num_valid, 2)
        self.assertEqual(config.model.value, "text-davinci-003")
        self.assertEqual(config.few_shot_prompt_type.value, "random")
        self.assertEqual(config.invalid_fn_type.value, "random")


if __name__ == "__main__":
    unittest.main()
