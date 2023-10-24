import random

from src.pipelines import ShotSamplingType
from src.pipelines.sequence_completions import (
    find_ambiguous_integer_sequences,
    sequence_functions,
)
from src.prompt_generation.robustness_checks.continuation_prompt import (
    create_continuation_prompt,
)


def ambiguous_sequences_test():
    ambiguous_sequences = find_ambiguous_integer_sequences(disambiguate=False)
    print(len(ambiguous_sequences))


def generate_random_fn_sequence_test():
    recursive_template = sequence_functions["recursive_progression"]
    recursive_fn = recursive_template.format(230, 100)
    print(eval(recursive_fn)(500))


def cont_prompt_test():
    for i in range(100):
        sequence_length = 10
        sequence = [random.randint(0, 100) for _ in range(sequence_length)]
        task_prompt = "self-consistency"
        model_name = "davinci"
        base = 10
        shots = 4
        shot_method = ShotSamplingType.RANDOM
        role_prompt = None
        seed = i
        continuation_prompt = create_continuation_prompt(
            sequence=sequence,
            task_prompt=task_prompt,
            role_prompt=role_prompt,
            model_name=model_name,
            base=base,
            shots=shots,
            shot_method=shot_method,
            seed=seed,
        )
        print(continuation_prompt)


def create_explanation_prompt_test():
    texto = create_continuation_prompt(
        sequence=[1, 2, 3, 4, 5],
        task_prompt="self-consistency",
        role_prompt=None,
        model_name="davinci",
        base=10,
        shots=4,
        shot_method=ShotSamplingType.RANDOM,
        seed=0,
    )
    print(texto)


if __name__ == "__main__":
    # ambiguous_sequences_test()
    # generate_random_fn_sequence_test()
    # cont_prompt_test()
    create_explanation_prompt_test()
