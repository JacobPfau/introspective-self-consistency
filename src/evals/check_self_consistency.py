from logging import getLogger
from typing import List, Optional

from src.evals.evaluate_continuation import generate_continuation, valid_continuation
from src.evals.evaluate_explanation import (
    generate_explanation,
    generate_implied_continuation,
    generate_implied_sequence,
    valid_explanation,
)
from src.prompt_generation.robustness_checks.continuation_prompt import (
    create_continuation_prompt,
)
from src.prompt_generation.robustness_checks.explanation_prompt import (
    create_explanation_prompt,
    parse_explanation,
)

logger = getLogger(__name__)


def self_consistency_evaluation(
    model_name: str,
    sequence: List[int],
    task_prompt: str,
    base: int,
    shots: int,
    shot_method: str,
    temperature: float,
    samples: int,
    role_prompt: Optional[str] = None,
    seed: int = 0,
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
        task_prompt=task_prompt,
        role_prompt=role_prompt,
        model_name=model_name,
        base=base,
        shots=shots,
        shot_method=shot_method,
    )

    explanation_prompt = create_explanation_prompt(
        sequence=sequence,
        task_prompt=task_prompt,
        role_prompt=role_prompt,
        model_name=model_name,
        base=base,
        shots=shots,
        shot_method=shot_method,
    )

    # Make the sequence base 2 if necessary
    if base == 2:
        sequence = [bin(i) for i in sequence]

    for _ in range(samples):
        result = {
            "continuation prompt": continuation_prompt,
            "explanation prompt": explanation_prompt,
            "continuation": None,
            "explanation": None,
            "implied sequence": None,
            "implied continuation": None,
            "correct": None,
            "consistent": None,
            "invalid": True,
        }
        logger.info("Generating a continuation and explanation")
        # Generate a continuation
        original_continuation = generate_continuation(
            prompt=continuation_prompt,
            model_name=model_name,
            temperature=temperature,
        )
        logger.info(f"continuation: {original_continuation}")
        result["continuation"] = original_continuation
        # strip whitespace
        continuation = original_continuation.strip()

        if not valid_continuation(continuation, base):
            logger.info("invalid continuation: ", continuation)
            total_results.append(result)
            continue

        # Generate an explanation
        explanation = generate_explanation(
            prompt=explanation_prompt,
            model_name=model_name,
            temperature=temperature,
        )
        logger.info(f"explanation: {explanation}")
        result["explanation"] = explanation
        # Parse explanation
        try:
            fn = parse_explanation(explanation)
        except BaseException:
            logger.info(f"invalid explanation - couldn't parse: {explanation}")
            total_results.append(result)
            continue

        if not valid_explanation(fn, len(sequence)):
            logger.info(f"invalid explanation: {explanation}")
            total_results.append(result)
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

        result["implied sequence"] = implied_sequence
        result["implied continuation"] = implied_continuation

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

        # try:
        #     # check if the implied continuation is decimal as specified
        #     _ = int(implied_continuation)
        # except ValueError:
        #     logger.info(f"invalid implied continuation: {implied_continuation}")
        #     total_results.append(result)
        #     continue

        if str(continuation) == str(implied_continuation):
            consistent = True
        else:
            consistent = False

        result["consistent"] = consistent
        result["correct"] = correct
        result["invalid"] = False
        total_results.append(result)

    return total_results
