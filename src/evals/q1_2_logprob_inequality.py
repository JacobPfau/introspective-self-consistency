"""Script to evaluate (Q1.2) the dependency on compute by looking at log probabilities.
In this setting, we generate ambiguous sequences with two valid rules and N invalid rules,
and prompt the model for (1) completion, (2) explanation, (3) explanation conditioned on priming prompt.
We compute/obtain the log probabilities for each answer and evaluate the mass distribution.
"""

import copy
import logging
import os
import random
from typing import Any, Dict, List, Literal, Union

import pandas as pd
import tiktoken
from hydra.utils import get_original_cwd
from tqdm import tqdm

from src.evals.config import Q12LogprobInequalityConfig
from src.evals.utils import parse_function_and_model_from_csv
from src.models.base_model import BaseModel
from src.models.openai_model import (
    OpenAITextModels,
    generate_logprob_response_with_turns,
    generate_response_with_turns,
)
from src.models.utils import get_model_from_string
from src.pipelines.sequence_completions import (
    find_ambiguous_integer_sequences,
    generate_sequence_completion_prompt,
    generate_shot_pool,
    resolve_fn,
)

_MIN_LOGPROB = -100.0


logger = logging.getLogger(__name__)


def list_rindex(li, x, ignore_idx: List[int] = []):
    # Get the index of the last occurrence of x in list
    for i in reversed(range(len(li))):
        if li[i] == x and i not in ignore_idx:
            return i
    raise ValueError(f"'{x}' is not in list or is in ignore_idx")


def _get_logprob_from_response(
    model: OpenAITextModels,
    completion_string: str,
    tokens: List[str],
    token_logprobs: List[float],
) -> float:

    encoding = tiktoken.encoding_for_model(model.value)
    # obtain string sub-tokens to find index in tokens
    string_tokens = [
        encoding.decode([tkn]) for tkn in encoding.encode(completion_string)
    ]
    try:
        # get logprob for each token in completion string
        # ensure that repeated tokens are handled correctly, i.e.
        tkn_indices: List[int] = []
        for tkn in string_tokens:
            idx = list_rindex(tokens, tkn, tkn_indices)
            tkn_indices.append(idx)

        logprobs = [token_logprobs[idx] for idx in tkn_indices]
        logprob = sum(logprobs) / len(logprobs)
    except ValueError as e:
        logger.error(f"Completion not found, return min logprob: {repr(e)}")
        logprob = _MIN_LOGPROB

    return logprob


def run_q1_2_eval(
    config: Q12LogprobInequalityConfig,
):
    """Main function to run Q1.2 eval."""
    config.csv_input_path = os.path.join(get_original_cwd(), config.csv_input_path)

    # main function to run this eval which can be called from main.py
    logger.info("Prep data for Q1.2 eval.")
    logger.info(f"Config: {repr(config)}")
    amb_seqs, data = get_data_q1_2(config)
    results = []
    for entry in tqdm(data):
        try:
            # roll out valid fns to obtain valid completions
            model: BaseModel = entry["model"]
            sequence = entry["sequence"]
            org_func = entry["org_func"]
            valid_fns = entry["valid_fns"]
            valid_completions = entry["valid_completions"]
            invalid_fns = entry["invalid_fns"]
            invalid_completions = entry["invalid_completions"]

            # run eval for sequence completion
            completion_responses, possible_completions = _eval_sequence_completion(
                model,
                org_func,
                config.num_shots,
                config.cot,
                config.few_shot_prompt_type,
                amb_seqs,
                sequence,
                valid_completions,
                invalid_completions,
            )

            test_passing_completion = evaluate_logprob_inequality(completion_responses)

            # run eval for sequence explanation
            explanation_responses, possible_explanations = _eval_sequence_explanation(
                model,
                org_func,
                config.num_shots,
                config.cot,
                config.few_shot_prompt_type,
                amb_seqs,
                sequence,
                valid_fns,
                invalid_fns,
            )

            test_passing_explanation = evaluate_logprob_inequality(
                explanation_responses
            )
            logger.info(
                f"Given sequence '{sequence}': \
                    \n- Completion test was passed: {test_passing_completion} \
                    \n- Explanation test was passed: {test_passing_explanation}"
            )

            # determine whether valid completions/explanations are in possible completions/explanations response
            n_valid_compl_in_primed_resp = len(
                [c for c in valid_completions if str(c) in possible_completions]
            )
            n_invalid_compl_in_primed_resp = len(
                [c for c in invalid_completions if str(c) in possible_completions]
            )
            n_valid_expl_in_primed_resp = len(
                [e for e in valid_fns if e["fn"] in possible_explanations]
            )
            n_invalid_expl_in_primed_resp = len(
                [e for e in invalid_fns if e["fn"] in possible_explanations]
            )

            # compose results entry
            results_entry = {
                "model": model.value,
                "sequence": sequence,
                "org_func": org_func,
                "num_valid": config.num_valid,
                "num_invalid": config.num_invalid,
                "num_shots": config.num_shots,
                "invalid_fn_type": config.invalid_fn_type,
                "test_passing_completion": int(test_passing_completion),
                "test_passing_explanation": int(test_passing_explanation),
                "n_valid_compl_in_primed_resp": n_valid_compl_in_primed_resp,
                "n_invalid_compl_in_primed_resp": n_invalid_compl_in_primed_resp,
                "n_valid_expl_in_primed_resp": n_valid_expl_in_primed_resp,
                "n_invalid_expl_in_primed_resp": n_invalid_expl_in_primed_resp,
                "possible_completions_response": possible_completions,
                "possible_explanations_response": possible_explanations,
            }

            results.append(results_entry)

        except Exception as e:
            logger.error(f"Unexpected error in Q1.2 eval: {repr(e)}")

    _save_results_to_csv(results)


def _save_results_to_csv(results: List[Dict[str, Any]]):
    df = pd.DataFrame.from_dict(results, orient="columns")

    # append to existing csv if exists
    csv_path = "results.csv"
    if os.path.exists(csv_path):
        df = pd.concat([pd.read_csv(csv_path, sep=","), df], ignore_index=True)

    df.to_csv(csv_path, sep=",", index=False, header=True)
    logger.info(f"Saved results to: {csv_path}.")


def get_data_q1_2(config: Q12LogprobInequalityConfig):
    """Based on consistent function determined in Q0, generate data samples for Q1.2.
    Each sample consists of:
        - ambiguous sequence
        - original function/explanation (i.e. the one determined in Q0)
        - valid alternative explanations
        - invalid alternative explanations
        - valid completions
        - invalid completions (based on the invalid explanations)

    """
    base_data = parse_function_and_model_from_csv(config.csv_input_path)

    # filter data to only include text models
    base_data = [
        entry
        for entry in base_data
        if get_model_from_string(entry["model"]) == OpenAITextModels.TEXT_DAVINCI_003
        # isinstance(get_model_from_string(entry["model"]), OpenAITextModels)
    ]  #
    logger.info("Skipping non-text models as logprobs are not available.")

    amb_seqs = find_ambiguous_integer_sequences()

    data = []

    for entry in tqdm(base_data, desc="Generating data for Q1.2 eval."):
        model = get_model_from_string(entry["model"])
        consistent_func = entry["fn_item"]
        # {'fn': 'lambda x: (1 * x) ** 1', 'offset': 0, 'metadata': ('exponential_progression', 0, 1)}

        # generate dataset for this eval:
        # 1) generate ambiguous sequence given a valid explanation and find alternative, valid explanation
        # 2) generate valid completions
        # 3) generate shots for invalid explanations
        # 4) generate invalid completions

        # find alternative, valid function
        for sequence, fns in list(amb_seqs.items()):
            entry = {}
            if consistent_func in fns:
                # sample valid ambiguous functions
                if len(fns) >= config.num_valid:
                    valid_fns = random.sample(fns, config.num_valid)
                    while consistent_func not in valid_fns:
                        valid_fns = random.sample(fns, config.num_valid)
                else:
                    logger.error("Not enough valid candidate functions.")
                    continue

                # roll out valid fns to obtain valid completions
                last_step = len(sequence.split(","))
                valid_completions = [
                    resolve_fn(fn_item, last_step) for fn_item in valid_fns
                ]

                # generate shots for alternative, invalid explanations
                invalid_fns = generate_shot_pool(
                    n_shots=config.num_invalid,
                    base_fn=consistent_func,
                    shot_type=config.invalid_fn_type,
                    ambiguous_sequences=amb_seqs,
                )
                invalid_completions = [
                    resolve_fn(fn_item, last_step) for fn_item in invalid_fns
                ]

                entry["sequence"] = sequence
                entry["org_func"] = consistent_func
                valids = list(zip(valid_fns, valid_completions))
                random.shuffle(valids)
                valid_fns, valid_completions = zip(*valids)

                entry["valid_fns"] = list(valid_fns)
                entry["valid_completions"] = valid_completions
                entry["invalid_fns"] = list(invalid_fns)
                entry["invalid_completions"] = invalid_completions
                entry["model"] = model

                data.append(entry)

    return amb_seqs, data


def _eval_sequence_completion(
    model: BaseModel,
    org_func: Dict[str, Any],
    num_shots: int,
    cot: bool,
    few_shot_prompt_type: str,
    amb_seqs: Dict[str, List[Dict[str, Union[str, int]]]],
    sequence: str,
    valid_completions: List[int],
    invalid_completions: List[int],
):
    completion_prompt = generate_sequence_completion_prompt(
        sequence,
        org_func,
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
        turns = copy.deepcopy(completion_prompt["prompt_turns"])
        completion_string = " " + str(completion)
        turns[-1]["content"] += completion_string

        tokens, token_logprobs = generate_logprob_response_with_turns(
            model,
            turns=turns,
            max_tokens=0,
        )

        logprob = _get_logprob_from_response(
            model, completion_string, tokens, token_logprobs
        )

        completion_responses.append(
            {"completion": completion, "logprob": logprob, "valid": valid}
        )

    turns = completion_prompt["prompt_turns"]

    # modify the last turn to ask for possible explanations
    priming_prompt = get_model_priming_prompt_possible_options(
        model, turns[-1], "completion"
    )
    turns[-1] = priming_prompt

    response = generate_response_with_turns(
        model, turns=turns
    )  # completions based on priming the model
    possible_completions = [
        elem.strip() for elem in response.split("\\n")
    ]  # parse response

    return completion_responses, possible_completions


def _eval_sequence_explanation(
    model: BaseModel,
    org_func: Dict[str, Any],
    num_shots: int,
    cot: bool,
    few_shot_prompt_type: str,
    amb_seqs: Dict[str, List[Dict[str, Union[str, int]]]],
    sequence: str,
    valid_fns: List[Dict[str, Any]],
    invalid_fns: List[Dict[str, Any]],
):
    # construct prompt for model explanation
    explanation_prompt = generate_sequence_completion_prompt(
        sequence,
        org_func,
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
        # turns[-1] = {'role': 'user', 'content': '\nFor the sequence: 0,2,4,6\n
        #   \nGive the code that generates the above sequence.\n'}
        turns = copy.deepcopy(explanation_prompt["prompt_turns"])
        explanation_string = " " + str(explanation["fn"])
        turns[-1]["content"] += explanation_string

        tokens, token_logprobs = generate_logprob_response_with_turns(
            model,
            turns=turns,
            max_tokens=0,
        )

        logprob = _get_logprob_from_response(
            model, explanation_string, tokens, token_logprobs
        )
        explanation_responses.append(
            {"completion": explanation, "logprob": logprob, "valid": valid}
        )

    turns = explanation_prompt["prompt_turns"]

    # modify the last turn to ask for possible explanations
    priming_prompt = get_model_priming_prompt_possible_options(
        model, turns[-1], "explanation"
    )
    turns[-1] = priming_prompt

    response = generate_response_with_turns(
        model, turns=turns
    )  # explanations based on priming the model
    possible_explanations = [elem.strip() for elem in response.split("\\n")]

    return explanation_responses, possible_explanations


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


def get_model_priming_prompt_possible_options(
    model: BaseModel,
    turn: Dict[str, str],
    response_type: Literal["completion", "explanation"],
) -> Dict[str, str]:
    # generate priming prompt for model
    base = f"Please list all possible {response_type}s separated by escape character '\\n' "
    model_string = f", as determined by you, {model.value}."
    seq = turn["content"].split("\n")[1]  # start with "For the sequence .."
    if response_type == "completion":
        priming = base + "which could be valid continuations of the sequence"

    elif response_type == "explanation":
        base = f"Please list all possible {response_type}s (as code) separated by escape character '\\n' "
        priming = base + "which could have generated the sequence above"

    turn["content"] = "\n" + seq + "\n\n" + priming + model_string + "\n"

    return turn
