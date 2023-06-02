from src.evals.q1_2_logprob_inequality import list_rindex


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
