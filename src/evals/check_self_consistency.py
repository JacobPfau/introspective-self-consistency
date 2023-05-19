from typing import List

from src.evals.evaluate_continuation import generate_continuation, valid_continuation
from src.evals.evaluate_explanation import (
    generate_explanation,
    generate_implied_continuation,
    generate_implied_sequence,
    valid_explanation,
)
from src.evals.prompts.continuation_prompt import create_continuation_prompt
from src.evals.prompts.explanation_prompt import (
    create_explanation_prompt,
    parse_explanation,
)


def self_consistency_evaluation(
    model_name: str,
    sequence: List[int],
    distribution: str,
    base: int,
    shots: int,
    shot_method: str,
    temperature: float,
    samples: int,
):
    """
    Given a sequence, prompt the model to both continue the sequence and
    generate an explanation for the sequence (via a python function). Compare
    whether the two outputs are consistent.
    """

    correct_consistent_explanations = 0
    correct_inconsistent_explanations = 0
    incorrect_consistent_explanations = 0
    incorrect_inconsistent_explanations = 0
    invalid_responses = 0

    # Generate a prompt
    continuation_prompt = create_continuation_prompt(
        sequence=sequence,
        distribution=distribution,
        model_name=model_name,
        base=base,
        shots=shots,
        shot_method=shot_method,
    )

    explanation_prompt = create_explanation_prompt(
        sequence=sequence,
        distribution=distribution,
        model_name=model_name,
        base=base,
        shots=shots,
        shot_method=shot_method,
    )

    for _ in range(samples):
        print("eyo")
        # Generate a continuation
        continuation = generate_continuation(
            prompt=continuation_prompt,
            model_name=model_name,
            temperature=temperature,
        )
        # strip whitespace
        continuation = continuation.strip()

        if not valid_continuation(continuation, base):
            print("invalid continuation: ", continuation)
            invalid_responses += 1
            continue
        if base == 2:
            continuation = int(continuation, 2)

        # Generate an explanation
        explanation = generate_explanation(
            prompt=explanation_prompt,
            model_name=model_name,
            temperature=temperature,
        )

        # Parse explanation
        try:
            fn = parse_explanation(explanation)
        except BaseException:
            invalid_responses += 1
            continue

        if not valid_explanation(fn, len(sequence)):
            print("invalid explanation: ", explanation)
            invalid_responses += 1
            continue
        else:
            # check if the explanation is valid up to the continuation
            implied_sequence = generate_implied_sequence(
                fn_form=fn,
                sequence_length=len(sequence),
            )

            implied_continuation = generate_implied_continuation(
                fn_form=fn,
                sequence_length=len(sequence),
            )

        # Check the explanation is accurate
        print("implied_sequence: ", implied_sequence)
        print("sequence: ", sequence)
        if implied_sequence == sequence:
            correct = True
        else:
            correct = False

        # Check consistency
        print("implied_continuation: ", implied_continuation)
        print("continuation: ", continuation)
        try:
            # check if the implied continuation is decimal as specified
            _ = int(implied_continuation)
        except ValueError:
            invalid_responses += 1
            continue

        if continuation == int(implied_continuation):
            consistent = True
        else:
            consistent = False

        if correct and consistent:
            correct_consistent_explanations += 1
        elif correct and not consistent:
            correct_inconsistent_explanations += 1
        elif not correct and consistent:
            incorrect_consistent_explanations += 1
        elif not correct and not consistent:
            incorrect_inconsistent_explanations += 1

    return (
        correct_consistent_explanations,
        correct_inconsistent_explanations,
        incorrect_consistent_explanations,
        incorrect_inconsistent_explanations,
        invalid_responses,
    )
