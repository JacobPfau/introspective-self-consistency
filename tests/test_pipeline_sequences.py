import unittest

from src.pipelines.sequences import (
    IntegerSequence,
    get_all_sequences,
    get_sequences_as_dict,
)

# Integer sequence functions
_ORIGINALSEQUENCE_FUNCTIONS = {
    "arithmetic": "lambda x: ({} * x) + {}",
    "geometric": "lambda x: ({} * x) * {}",
    "exponential": "lambda x: ({} * x) ** {}",
    "power": "lambda x: {} ** ({} * x)",
    "bitwise_or": "lambda x: ({} * x) | {}",
    "modular": "lambda x: (x * {}) % ({}+1)",
    "indexing_criteria": (
        "lambda x: [i for i in range(100) if i % ({} + 1) or i % ({} + 1)][x]"
    ),
    "recursive": (
        "(lambda a:lambda v:a(a,v))(lambda fn,x:1 if x==0 else {} * x * fn(fn,x-1) + {})"
    ),
}


class TestSequences(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = None

    def test_get_sequence_dict_contains_all_sequences(self):
        seq_dict = get_sequences_as_dict()
        self.assertDictEqual(seq_dict, _ORIGINALSEQUENCE_FUNCTIONS)

    def test_integer_sequence_rolls_out_correctly(self):
        seqs = get_all_sequences()

        for seq in seqs:
            for offset in [1, 2, 3]:
                for term_a in [1, 2, 3]:
                    for term_b in [1, 2, 3]:

                        seq_obj = IntegerSequence(
                            seq, offset=offset, term_a=term_a, term_b=term_b
                        )
                        self.assertEqual(seq_obj.sequence_type, seq)

                        base_fn = seq.base_fn.format(term_a, term_b)
                        self.assertEqual(str(seq_obj), base_fn)

                        # manula roll out of the function
                        for x in range(10):
                            self.assertEqual(
                                seq_obj.roll_out(x), eval(base_fn)(x + offset)
                            )


if __name__ == "__main__":
    unittest.main()
