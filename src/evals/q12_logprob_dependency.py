"""Script to evaluate (Q1.2) the dependency on compute by looking at log probabilities.
In this setting, we generate ambiguous sequences with two valid rules and N invalid rules,
and prompt the model for (1) completion, (2) explanation, (3) explanation conditioned on priming prompt.
We compute/obtain the log probabilities for each answer and evaluate the mass distribution.
"""

import logging
import random
from typing import Any, Dict, List, Literal, Optional, Union

from tqdm import tqdm

from src.models.openai_model import (
    OpenAITextModels,
    generate_logprob_response_with_turns,
)
from src.models.utils import BaseModel
from src.pipelines.sequence_completions import (
    _generate_shot_pool,
    _resolve_fn,
    find_ambiguous_integer_sequences,
    generate_sequence_completion_prompt,
)

_RESPONSE_TYPES = ["completion", "explanation"]

_MIN_LOGPROB = -100.0


logger = logging.getLogger(__name__)


def list_rindex(li, x):
    # Get the index of the last occurrence of x in list
    for i in reversed(range(len(li))):
        if li[i] == x:
            return i
    raise ValueError(f"'{x}' is not in list")


def _get_logprob_from_response(
    completion_string: str, tokens: List[str], token_logprobs: List[float]
) -> float:
    try:
        logprob = token_logprobs[list_rindex(tokens, completion_string)]
    except ValueError as e:
        logger.error(f"Completion not found, return min logprob: {repr(e)}")
        logprob = _MIN_LOGPROB

    return logprob


def run_q1_2_eval(
    model: BaseModel,
    max_offset=8,
    num_shots=4,
    num_valid=2,
    num_invalid=3,
    cot=False,
    few_shot_prompt_type="random",
    invalid_fn_type="random",
):

    # main function to run this eval which can be called from main.py

    # generate ambiguous sequences
    # first find amb seqs with 2 possible explanations (generation functions/rules)
    amb_seqs = find_ambiguous_integer_sequences()

    for sequence, fns in tqdm(list(amb_seqs.items())):
        valid_fns = random.sample(fns, num_valid) if len(fns) >= num_valid else fns
        try:
            # roll out valid fns to obtain valid completions
            last_step = len(sequence.split(","))
            valid_completions = [
                _resolve_fn(fn_item, last_step) for fn_item in valid_fns
            ]

            # generate shots for alternative, invalid explanations
            invalid_fns = _generate_shot_pool(
                n_shots=num_invalid,
                base_fn=valid_fns[0],
                shot_type=invalid_fn_type,
                ambiguous_sequences=amb_seqs,
            )  # TODO: decide which function to use
            invalid_completions = [
                _resolve_fn(fn_item, last_step) for fn_item in invalid_fns
            ]

            # run eval for sequence completion
            # completion_responses = eval_sequence_completion(
            #     model,
            #     num_shots,
            #     cot,
            #     few_shot_prompt_type,
            #     amb_seqs,
            #     sequence,
            #     valid_fns,
            #     valid_completions,
            #     invalid_completions,
            # )

            # test_passing_completion = evaluate_logprob_inequality(completion_responses)
            # print(f"Given sequence '{sequence}', completion test was passed: {test_passing_completion}")

            # sort completion responses by logprob with largest first
            # completion_responses = sorted(completion_responses, key=lambda x: x["logprob"], reverse=True)
            # print(completion_responses)

            ############################
            # run eval for sequence explanation

            # construct prompt for model explanation
            explanation_prompt = generate_sequence_completion_prompt(
                sequence,
                valid_fns[0],
                prompt_type="explanation",
                n_shots=num_shots,
                use_cot=cot,
                ambiguous_sequences=amb_seqs,
                shot_type=few_shot_prompt_type,
            )

            # prompt model for explanation and obtain log probabilities for each response
            explanation_responses = []

            for explanation, valid in [(expl, True) for expl in valid_fns] + [
                (expl, False) for expl in invalid_fns
            ]:
                # add explanation to the last prompt turn
                # turns[-2] = {'role': 'assistant', 'content': 'lambda x: 4 ** (3 * x)'}
                # turns[-1] = {'role': 'user', 'content': '\nFor the sequence: 0,2,4,6\n\nGive the code that generates the above sequence.\n'}
                turns = explanation_prompt["prompt_turns"]
                explanation_string = " " + str(explanation)
                turns[-1]["content"] += explanation_string

                # TODO: how to make infernece more efficient? currently using all turns for every completion request
                tokens, token_logprobs = generate_logprob_response_with_turns(
                    model,
                    turns=turns,
                    max_tokens=0,
                )

                # TODO: handle multi-token completions
                logprob = _get_logprob_from_response(
                    explanation_string, tokens, token_logprobs
                )
                explanation_responses.append(
                    {"completion": explanation, "logprob": logprob, "valid": valid}
                )

        except Exception as e:
            print(e)
    # TODO: extend to

    # construct prompt for model explanation conditioned on priming prompt

    # prompt model for completion/explanation/condiationed explanation

    # obtain log probabilities for each response

    # evaluate test-passing condition

    pass


def eval_sequence_completion(
    model,
    num_shots: int,
    cot: bool,
    few_shot_prompt_type: str,
    amb_seqs: Dict[str, List[Dict[str, Union[str, int]]]],
    sequence: str,
    valid_fns: List[Dict[str, Any]],
    valid_completions: List[int],
    invalid_completions: List[int],
    priming_prompt: Optional[str] = None,
):
    completion_prompt = generate_sequence_completion_prompt(
        sequence,
        valid_fns[0],
        n_shots=num_shots,
        use_cot=cot,
        ambiguous_sequences=amb_seqs,
        shot_type=few_shot_prompt_type,
    )

    # prompt model for completion and obtain log probabilities for each response
    completion_responses = []

    for completion, valid in [(compl, True) for compl in valid_completions] + [
        (compl, False) for compl in invalid_completions
    ]:
        # add completion to the last prompt turn
        turns = completion_prompt["prompt_turns"]
        completion_string = " " + str(completion)

        if priming_prompt is not None:
            turns.append({"speaker": "user", "content": priming_prompt + completion_string})
        else:
            turns[-1]["content"] += completion_string

        # TODO: how to make infernece more efficient? currently using all turns for every completion request
        tokens, token_logprobs = generate_logprob_response_with_turns(
            model,
            turns=turns,
            max_tokens=0,
        )

        # TODO: handle multi-token completions
        logprob = _get_logprob_from_response(completion_string, tokens, token_logprobs)

        completion_responses.append(
            {"completion": completion, "logprob": logprob, "valid": valid}
        )

    return completion_responses


def evaluate_logprob_inequality(
    completion_responses: List[Dict[str, Any]],
) -> bool:
    # If P(e_valid_i) > P(e_invalid) & P(c_valid_i) > P(c_invalid) hold for all e_invalid, c_invalid

    # separate valid and invalid responses
    logprobs_valid = [
        response["logprob"] for response in completion_responses if response["valid"]
    ]
    logprobs_invalid = [
        response["logprob"]
        for response in completion_responses
        if not response["valid"]
    ]

    ineq = []

    for valid_prob in logprobs_valid:
        ineq.append(
            all([valid_prob > invalid_prob for invalid_prob in logprobs_invalid])
        )

    return all(ineq)


def get_model_priming_prompt(
    model: BaseModel, response_type: Literal["completion", "explanation"]
) -> str:
    # generate priming prompt for model
    # TODO: all possible? Or single best?
    priming = f"Please list all possible {response_type}s which could have generated the data, as determined by you, {model.value}."
    return priming


def determine_threshold_for_test_passing(logprobs: Dict[str, float]):
    # empirically determine threshold for test-passing behaviour:
    # Which value of __P(e_valid_i | 3.5)__ under __INEQ(e_valid_i)__ is considered high?
    pass


def list_logprob_distribution(
    logprobs_valid: Dict[str, float],
    lobprobs_invalid: Dict[str, float],
    response_type: Optional[str] = None,
) -> List[float]:
    # simple list of logprob over responses, ordered by  for valid and invalid responses

    pass


if __name__ == "__main__":

    model = OpenAITextModels.TEXT_DAVINCI_003

    # # demonstration of prompting with multiple choice for log probs
    # _, log_probs_dog = generate_logprob_response_with_turns(
    #     model, [
    #         {"content": "Q: What is cuter A: Cats or B: Dogs? A: Dogs"},
    #     ],
    #     max_tokens=0
    # )
    # _, log_probs_cat = generate_logprob_response_with_turns(
    #     model, [
    #         {"content": "Q: What is cuter A: Cats or B: Dogs? A: Dogs"},
    #     ],
    #     max_tokens=0
    # )
    # print(log_probs_dog)
    # print(log_probs_cat)

    run_q1_2_eval(model)
