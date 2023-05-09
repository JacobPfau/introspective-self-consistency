from typing import List, Union

from models.openai_model import CHAT_MODEL_NAME, DAVINCI_MODEL_NAME

from q11.prompts.continuation_prompt import create_continuation_prompt
from q11.prompts.explanation_prompt import create_explanation_prompt, parse_explanation
from q11.evals.evaluate_continuation import valid_continuation, generate_continuation
from q11.evals.evaluate_explanation import valid_explanation, generate_explanation, generate_implied_continuation


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

    consistent_explanations = 0
    inconsistent_explanations = 0
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

        if not valid_continuation(continuation):
            invalid_responses += 1
            continue
        else:
            int_response = int(continuation)
        

        # Generate an explanation
        explanation = generate_explanation(
            prompt=explanation_prompt,
            model_name=model_name,
            temperature=temperature,
        )

        # Parse explanation
        try:
            fn, offset = parse_explanation(explanation)
        except:
            invalid_responses += 1
            continue

        offset = int(offset)

        if not valid_explanation(fn, offset, len(sequence)):
            invalid_responses += 1
            continue
        else:
            implied_continuation = generate_implied_continuation(
                fn_form=fn,
                offset=offset,
                sequence_length=len(sequence),
            )

        # Check consistency
        print("implied_continuation: ", implied_continuation)
        print("continuation: ", continuation)
        if int(continuation) == int(implied_continuation):
            consistent_explanations += 1
        else:
            inconsistent_explanations += 1
    
    return consistent_explanations, inconsistent_explanations, invalid_responses

