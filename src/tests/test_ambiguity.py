from src.pipelines.sequence_completions import find_ambiguous_integer_sequences

def test_ambiguous_sequences():
    ambiguous_sequences = find_ambiguous_integer_sequences(
    max_constant_term_one = 5,
    max_constant_term_two = 5,
    num_steps_to_check = 2,
    step_offsets = 5,
    )
    print(list(ambiguous_sequences.keys()))
    print(len(ambiguous_sequences))


if __name__ == "__main__":
    test_ambiguous_sequences()
