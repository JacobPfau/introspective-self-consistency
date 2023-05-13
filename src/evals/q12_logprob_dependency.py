"""Script to evaluate (Q1.2) the dependency on compute by looking at log probabilities.
In this setting, we generate ambiguous sequences with two valid rules and N invalid rules,
and prompt the model for (1) completion, (2) explanation, (3) explanation conditioned on priming prompt.
We compute/obtain the log probabilities for each answer and evaluate the mass distribution.
"""

from typing import Dict, List, Literal, Optional

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


def run_q1_2_eval(
    model: BaseModel,
    max_offset=8,
    num_shots=4,
    num_invalid=3,
    cot=False,
    few_shot_prompt_type="random",
    invalid_fn_type="random",
):

    # main function to run this eval which can be called from main.py

    # generate ambiguous sequences
    # first find amb seqs with 2 possible explanations (generation functions/rules)
    amb_seqs = find_ambiguous_integer_sequences()
    n_seqs = sum(len(fns) for fns in amb_seqs.values())

    # generate shots for amb seq, i.e. alternative, invalid explanations

    # construct prompt for model completion
    for sequence, fns in tqdm(list(amb_seqs.items())):

        try:
            # roll out valid fns to obtain valid completions
            last_step = len(sequence.split(","))
            valid_completions = [_resolve_fn(fn_item, last_step) for fn_item in fns]

            # generate invalid explanations
            invalid_fns = _generate_shot_pool(
                n_shots=num_invalid,
                base_fn=fns[0],
                shot_type=invalid_fn_type,
                ambiguous_sequences=amb_seqs,
            )
            invalid_completions = [
                _resolve_fn(fn_item, last_step) for fn_item in invalid_fns
            ]

            completion_prompt = generate_sequence_completion_prompt(
                sequence,
                fns[0],
                n_shots=num_shots,
                use_cot=cot,
                ambiguous_sequences=amb_seqs,
                shot_type=few_shot_prompt_type,
            )

            # prompt model for completion and obtain log probabilities for each response
            completion_resp = generate_logprob_response_with_turns(
                model, completion_prompt["prompt_turns"]
            )

            # evaluate test-passing condition

        except Exception as e:
            print(e)
    # TODO: extend to
    # construct prompt for model explanation
    # construct prompt for model explanation conditioned on priming prompt

    # prompt model for completion/explanation/condiationed explanation

    # obtain log probabilities for each response

    # evaluate test-passing condition

    pass


def evaluate_logprob_inequality(
    logprobs_valid: Dict[str, float], lobprobs_invalid: Dict[str, float]
) -> bool:
    # If P(e_valid_i) > P(e_invalid) & P(c_valid_i) > P(c_invalid) hold for all e_invalid, c_invalid
    ineq = []

    for valid_response, valid_prob in logprobs_valid.items():
        ineq.append(
            all(
                [
                    valid_prob > invalid_prob
                    for invalid_prob in lobprobs_invalid.values()
                ]
            )
        )

    return all(ineq)


def generate_sequence_completion_prompt_with_multiple_choice(
    completion_candadites: List[int],
) -> str:
    mc_string = "\n".join(completion_candadites)
    prompt = f"What is your best guess for the next number in the sequence? Select your best guess from the following options:{mc_string}\n"

    return prompt


def get_model_priming_prompt(
    model: BaseModel, response_type: Literal["completion", "explanation"]
) -> str:
    # generate priming prompt for model
    priming = f"Please list all possible {response_type} which could have generated the data, as determined by you, {model.value}."
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

    run_q1_2_eval(model)
