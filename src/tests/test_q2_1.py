from unittest import TestCase

from src.evals.config import Q21LogprobInequalityConfig
from src.evals.q2_1_logprob_inequality import list_rindex


class TestQ21(TestCase):
    def test_list_rindex_with_unique_tokens(self):
        test_string = list("abcdefghijklmnopqrstuvwxyz")

        assert list_rindex(test_string, "z") == 25
        assert list_rindex(test_string, "a") == 0
        assert list_rindex(test_string, "b") == 1


def test_list_rindex_with_repeated_token():
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

    assert tkn_indices == [2, 6, 7, 1, 5]


def test_eval_config_from_dict_initializes_correctly():
    test_dict = {
        "task": "q1_2_logprob_inequality",
        "model": "text-davinci-003",
        "csv_input_path": "data/q1_2_logprob_inequality.csv",
        "num_shots": 13,
        "num_invalid": 1,
    }

    config = Q21LogprobInequalityConfig.from_dict(test_dict)

    assert config.task == "q1_2_logprob_inequality"
    assert config.num_shots == 13
    assert config.num_invalid == 1
    assert config.num_valid == 2


def test_generate_sequence_explanation_prompts_proper_mc_format():
    # check whether the generated sequence explanation prompts are in the
    # proper multiple choice format
    raise NotImplementedError()


if __name__ == "__main__":

    test_eval_config_from_dict_initializes_correctly()
