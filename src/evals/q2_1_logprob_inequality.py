"""Script to evaluate (Q2.1) to what extent the model is considering alternatives by looking at log probabilities.
In this setting, we generate ambiguous sequences with two valid rules and N invalid rules,
and prompt the model for completion and explanation.
We compute/obtain the log probabilities for each answer and evaluate the mass distribution.
"""

import copy
import logging
import os
import random
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import tiktoken
from hydra.utils import get_original_cwd
from tqdm import tqdm

from src.evals.config import Q21LogprobInequalityConfig
from src.models import BaseModel, OpenAITextModels
from src.models.openai_model import (
    generate_logprob_response_with_turns,
    generate_response_with_turns,
)
from src.pipelines.alternative_completions import get_data_with_alternatives
from src.pipelines.sequence_completions import (
    generate_sequence_completion_prompt,
    generate_sequence_explanation_prompt_with_multiple_choices,
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


def run_q2_1_eval(
    config: Q21LogprobInequalityConfig,
):
    """Main function to run Q2.1 eval."""
    config.csv_input_path = os.path.join(get_original_cwd(), config.csv_input_path)

    # main function to run this eval which can be called from main.py
    logger.info("Prep data for Q2.1 eval.")
    logger.info("Skipping non-text models as logprobs are not available.")
    amb_seqs, data = get_data_with_alternatives(
        config.csv_input_path,
        config.num_valid,
        config.num_invalid,
        config.invalid_fn_type,
        skip_non_text_models=True,
    )
    results = []
    logprob_results = []
    for entry in tqdm(data, desc="Evaluating Q2.1"):
        try:
            model: BaseModel = entry["model"]
            sequence = entry["sequence"]
            org_func = entry["org_func"]
            valid_fns = entry["valid_fns"]
            valid_completions = entry["valid_completions"]
            invalid_fns = entry["invalid_fns"]
            invalid_completions = entry["invalid_completions"]

            # run eval for sequence completion
            completion_responses = _eval_sequence_completion(
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
            explanation_responses = _eval_sequence_explanation(
                model,
                org_func,
                config.num_shots,
                config.num_multiple_choices,
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

            # compose results entry
            results_entry = {
                "model": model.value,
                "sequence": sequence,
                "org_func": org_func,
                "num_valid": config.num_valid,
                "num_invalid": config.num_invalid,
                "num_shots": config.num_shots,
                "num_mc": config.num_multiple_choices,
                "invalid_fn_type": config.invalid_fn_type,
                "test_passing_completion": int(test_passing_completion),
                "test_passing_explanation": int(test_passing_explanation),
            }

            results.append(results_entry)
            logprob_results = _add_logprob_entries(
                logprob_results,
                model,
                sequence,
                org_func,
                config,
                completion_responses,
                explanation_responses,
                valid_fns=valid_fns,
                valid_completions=valid_completions,
                invalid_fns=invalid_fns,
                invalid_completions=invalid_completions,
            )

        except Exception as e:
            logger.error(f"Unexpected error in Q2.1 eval: {repr(e)}")

    _save_results_to_csv(results)
    _save_results_to_csv(logprob_results, csv_path="logprobs.csv")


def _get_completion_value_from_row(row, response_type: Optional[str] = None) -> str:
    if response_type is None:
        response_type = row["response_type"]
    if "answer" in row:
        return row["answer"]
    else:
        raise NotImplementedError(f"Unknown response type: {response_type}")


def _get_valid_and_pred_entries(
    logprob_entry: dict, pred_val: str, valid_vals: List[Any], invalid_vals: List[Any]
) -> dict:
    """Determine variations of whether this entry was valid and predicted or not."""

    # update logprob entry
    if logprob_entry["valid"] in ["valid", "invalid"]:
        logprob_entry["valid_and_pred"] = (
            1
            if logprob_entry["valid"] == "valid"
            and _get_completion_value_from_row(logprob_entry) == pred_val
            else 0
        )
        logprob_entry["valid_and_not_pred"] = (
            1
            if logprob_entry["valid"] == "valid"
            and _get_completion_value_from_row(logprob_entry) != pred_val
            else 0
        )
        logprob_entry["invalid_and_pred"] = (
            1
            if logprob_entry["valid"] == "invalid"
            and _get_completion_value_from_row(logprob_entry) == pred_val
            else 0
        )
        logprob_entry["invalid_and_not_pred"] = (
            1
            if logprob_entry["valid"] == "invalid"
            and _get_completion_value_from_row(logprob_entry) != pred_val
            else 0
        )
    else:
        logprob_entry["valid_and_pred"] = (
            1 if logprob_entry["valid"] == "pred" and pred_val in valid_vals else 0
        )
        logprob_entry["valid_and_not_pred"] = 0
        logprob_entry["invalid_and_pred"] = (
            1 if logprob_entry["valid"] == "pred" and pred_val in invalid_vals else 0
        )
        logprob_entry["invalid_and_not_pred"] = 0

    return logprob_entry


def _add_logprob_entries(
    logprob_results: List[dict],
    model: BaseModel,
    sequence: str,
    org_func,
    config: Q21LogprobInequalityConfig,
    completion_responses: List[dict],
    explanation_responses: List[dict],
    valid_fns: List[dict],
    valid_completions: List[str],
    invalid_fns: List[dict],
    invalid_completions: List[str],
) -> List[dict]:
    """Add logprob entries to logprob_results."""

    for entry in completion_responses:
        if entry["valid"] == "pred":
            pred_val = _get_completion_value_from_row(entry, "completion")
            break

    for entry in completion_responses:
        compl, logprob, valid = entry.values()

        logprob_entry = {
            "model": model.value,
            "sequence": sequence,
            "org_func": org_func,
            "num_valid": config.num_valid,
            "num_invalid": config.num_invalid,
            "num_shots": config.num_shots,
            "num_mc": config.num_multiple_choices,
            "invalid_fn_type": config.invalid_fn_type,
            "response_type": "completion",
            "answer": compl,
            "logprob": logprob,
            "valid": valid,
        }

        logprob_entry = _get_valid_and_pred_entries(
            logprob_entry, pred_val, valid_completions, invalid_completions
        )

        logprob_results.append(logprob_entry)

    # Explanations
    # get pred val and valid options
    valid_options = []
    pred_val = None
    for entry in explanation_responses:
        if entry["valid"] == "valid":
            valid_options.append(entry["answer"])
        elif entry["valid"] == "pred":
            pred_val = str(entry["answer"])

    for entry in explanation_responses:
        expl, logprob, valid = entry.values()

        logprob_entry = {
            "model": model.value,
            "sequence": sequence,
            "org_func": org_func,
            "num_valid": config.num_valid,
            "num_invalid": config.num_invalid,
            "num_shots": config.num_shots,
            "num_mc": config.num_multiple_choices,
            "invalid_fn_type": config.invalid_fn_type,
            "response_type": "explanation",
            "answer": expl,
            "logprob": logprob,
            "valid": valid,
        }
        # check whether predicted value was valid or not
        logprob_entry = _get_valid_and_pred_entries(
            logprob_entry, pred_val, valid_options, invalid_vals=[]
        )
        logprob_results.append(logprob_entry)

    return logprob_results


def _save_results_to_csv(results: List[Dict[str, Any]], csv_path="results.csv"):
    df = pd.DataFrame.from_dict(results, orient="columns")

    # append to existing csv if exists
    if os.path.exists(csv_path):
        df = pd.concat([pd.read_csv(csv_path, sep=","), df], ignore_index=True)

    df.to_csv(csv_path, sep=",", index=False, header=True)
    logger.info(f"Saved results to: {csv_path}.")


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
    # 1)
    # prompt model for completion and obtain log probabilities for each response
    completion_responses = []
    for completion, valid in [(compl, "valid") for compl in valid_completions] + [
        (compl, "invalid") for compl in invalid_completions
    ]:
        # add completion to the last prompt turn
        turns = copy.deepcopy(completion_prompt["prompt_turns"])
        completion_string = " " + str(completion)

        turns.append(
            {
                "role": "assistant",
                "content": completion_string,
            }
        )

        tokens, token_logprobs = generate_logprob_response_with_turns(
            model,
            turns=turns,
            max_tokens=0,
        )

        logprob = _get_logprob_from_response(
            model, completion_string, tokens, token_logprobs
        )

        completion_responses.append(
            {"answer": completion, "logprob": logprob, "valid": valid}
        )

    # 2)
    # get prediction for the ambiguous sequence
    pred_completion = generate_response_with_turns(
        model, turns=completion_prompt["prompt_turns"]
    ).strip()

    # parse predicted completion
    try:
        for pred_piece in pred_completion.split("\n"):
            if pred_piece != "":
                pred_completion = int(pred_piece.strip())
                break
    except ValueError:
        logger.warning(f"Could not parse predicted completion: {pred_completion}")

    # get logprob for the predicted completion
    if pred_completion in valid_completions:
        for compl_resp in completion_responses:
            if compl_resp["answer"] == pred_completion:
                pred_logprob = compl_resp["logprob"]
                break
    else:
        # add predicted completion as the last prompt turn
        turns = copy.deepcopy(completion_prompt["prompt_turns"])
        completion_string = " " + str(pred_completion)

        turns.append(
            {
                "role": "assistant",
                "content": completion_string,
            }
        )

        tokens, token_logprobs = generate_logprob_response_with_turns(
            model,
            turns=turns,
            max_tokens=0,
        )

        pred_logprob = _get_logprob_from_response(
            model, completion_string, tokens, token_logprobs
        )

    completion_responses.append(
        {"answer": pred_completion, "logprob": pred_logprob, "valid": "pred"}
    )

    return completion_responses


def _eval_sequence_explanation(
    model: BaseModel,
    org_func: Dict[str, Any],
    n_shots: int,
    n_mc_options: int,
    few_shot_prompt_type: str,
    amb_seqs: Dict[str, List[Dict[str, Union[str, int]]]],
    sequence: str,
    valid_fns: List[Dict[str, Any]],
    invalid_fns: List[Dict[str, Any]],
):
    # construct prompt for model explanation with in-context demonstrations of multiple choices
    explanation_prompt = generate_sequence_explanation_prompt_with_multiple_choices(
        sequence,
        org_func,
        model,
        n_shots=n_shots,
        n_mc_options=n_mc_options,
        ambiguous_sequences=amb_seqs,
        shot_type=few_shot_prompt_type,
    )

    # 1)
    # prompt model for explanation and obtain log probabilities for each response
    multi_choice_options = [(expl, "valid") for expl in valid_fns] + [
        (expl, "invalid") for expl in invalid_fns
    ]
    random.shuffle(multi_choice_options)  # shuffle options to avoid bias

    # construct multiple choice prompt
    mc_prompt = ""
    for i, (expl, _) in enumerate(multi_choice_options):
        mc_prompt += f"{i+1}. {expl['fn']}\n"

    # add mc options of explanation to the last prompt turn
    explanation_prompt["prompt_turns"][-1]["content"] += mc_prompt

    explanation_responses = []
    for i, (_, valid) in enumerate(multi_choice_options):
        turns = copy.deepcopy(explanation_prompt["prompt_turns"])

        # add number of option as the last prompt turn
        choice_string = str(i + 1)
        turns.append(
            {
                "role": "assistant",
                "content": choice_string,
            }
        )

        tokens, token_logprobs = generate_logprob_response_with_turns(
            model,
            turns=turns,
            max_tokens=0,
        )

        logprob = _get_logprob_from_response(
            model, choice_string, tokens, token_logprobs
        )
        explanation_responses.append(
            {
                "answer": choice_string,
                "logprob": logprob,
                "valid": valid,
            }
        )

    # 2)
    # get prediction for the ambiguous sequence
    turns = copy.deepcopy(explanation_prompt["prompt_turns"])  # ends with mc options
    pred_explanation = generate_response_with_turns(model, turns=turns).strip()

    # since model is primed with multiple choice, the predicted explanation must be a number and part of the explanation options
    try:
        pred_parsed = (
            pred_explanation.split("\n")[0].split(" ")[0].replace(".", "").strip()
        )  # number should be the first token, perhaps with a dot
        pred_parsed = int(pred_parsed)
        # re-use logprob from previous response if that was the predicted explanation
        pred_logprob = explanation_responses[pred_parsed - 1]["logprob"]

    except (ValueError, IndexError) as e:
        logger.warning(
            f"Could not parse integer from predicted explanation and will explicitly. {repr(e)}"
        )
        pred_parsed = (
            pred_explanation.split("\n")[0].split(" ")[0].replace(".", "").strip()
        )
        turns.append(
            {
                "role": "assistant",
                "content": pred_parsed,
            }
        )

        tokens, token_logprobs = generate_logprob_response_with_turns(
            model,
            turns=turns,
            max_tokens=0,
        )

        pred_logprob = _get_logprob_from_response(
            model, pred_parsed, tokens, token_logprobs
        )

    explanation_responses.append(
        {
            "answer": pred_parsed,
            "logprob": pred_logprob,
            "valid": "pred",
        }
    )

    return explanation_responses


def evaluate_logprob_inequality(
    completion_responses: List[Dict[str, Any]],
) -> bool:
    # If P(e_valid_i) > P(e_invalid) & P(c_valid_i) > P(c_invalid) hold for all e_invalid, c_invalid

    # separate valid and invalid responses
    logprobs_valid = [
        response["logprob"]
        for response in completion_responses
        if response["valid"] == "valid"
    ]
    logprobs_invalid = [
        response["logprob"]
        for response in completion_responses
        if response["valid"] == "invalid"
    ]

    ineq = []

    for valid_prob in logprobs_valid:
        ineq.append(
            all([valid_prob > invalid_prob for invalid_prob in logprobs_invalid])
        )

    return all(ineq)


if __name__ == "__main__":

    config = Q21LogprobInequalityConfig(
        task="q2_1_inequality",
        model="text-davinci-003",
        csv_input_path="data/q2_functions/consistent_functions_by_model.csv",
    )

    run_q2_1_eval(config)
