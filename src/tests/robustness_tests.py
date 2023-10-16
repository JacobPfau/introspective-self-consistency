from src.pipelines.sequence_completions import find_ambiguous_integer_sequences


def ambiguous_sequences_test():
    ambiguous_sequences = find_ambiguous_integer_sequences(disambiguate=False)
    print(len(ambiguous_sequences))


if __name__ == "__main__":
    ambiguous_sequences_test()
