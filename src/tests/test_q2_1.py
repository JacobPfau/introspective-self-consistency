from src.evals.config import Q12LogprobInequalityConfig
from src.evals.q2_1_logprob_inequality import list_rindex


def test_list_rindex_with_unique_tokens():
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

    config = Q12LogprobInequalityConfig.from_dict(test_dict)

    assert config.task == "q1_2_logprob_inequality"
    assert config.num_shots == 13
    assert config.num_invalid == 1
    assert config.num_valid == 2


if __name__ == "__main__":

    test_eval_config_from_dict_initializes_correctly()
