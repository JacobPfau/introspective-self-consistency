import logging
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

logger = logging.getLogger(__name__)


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

    total_results = []

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
        invalid_response = True
        logger.info("Generating a continuation and explanation")
        # Generate a continuation
        original_continuation = generate_continuation(
            prompt=continuation_prompt,
            model_name=model_name,
            temperature=temperature,
        )
        # strip whitespace
        continuation = original_continuation.strip()

        if not valid_continuation(continuation, base):
            logger.info("invalid continuation: ", continuation)
            invalid_response = True
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
            invalid_response = True
            continue

        if not valid_explanation(fn, len(sequence)):
            logger.info(f"invalid explanation: {explanation}")
            invalid_response = True
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
        logger.info(f"implied_sequence: {implied_sequence}")
        logger.info(f"sequence: {sequence}")
        if implied_sequence == sequence:
            correct = True
        else:
            correct = False

        # Check consistency
        logger.info(f"implied_continuation: {implied_continuation}")
        logger.info(f"continuation: {continuation}")
        try:
            # check if the implied continuation is decimal as specified
            _ = int(implied_continuation)
        except ValueError:
            invalid_response = True
            continue

        if int(continuation) == int(implied_continuation):
            consistent = True
        else:
            consistent = False

        single_result = {
            "continuation prompt": continuation_prompt,
            "explanation prompt": explanation_prompt,
            "continuation": original_continuation,
            "explanation": explanation,
            "implied sequence": implied_sequence,
            "implied continuation": implied_continuation,
            "correct": correct,
            "consistent": consistent,
            "invalid": invalid_response,
        }

        total_results.append(single_result)

    return total_results
