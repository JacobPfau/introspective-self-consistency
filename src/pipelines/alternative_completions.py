import ast
import csv
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from src.models import BaseModel, OpenAITextModels, get_model_from_string
from src.pipelines.sequence_completions import (
    find_ambiguous_integer_sequences,
    generate_shot_pool,
    resolve_fn,
)

logger = logging.getLogger(__name__)


def parse_function_and_model_from_csv(
    csv_path: str, delimiter=","
) -> List[Dict[str, Any]]:
    """
    Parse the function and model from a csv file.
    """
    data = []

    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        for row in csv_reader:
            if row[0] == "":
                continue
            entry = {}
            entry["fn_item"] = ast.literal_eval(row[1])  # convert string to dict
            entry["model"] = row[2]
            data.append(entry)

    return data


def _generate_invalid_alternatives(
    num_invalid,
    invalid_fn_type,
    org_fn,
    last_step: int,
    amb_seqs,
    all_valid_fns,
    valid_completions,
) -> Tuple[List[dict], List[int]]:
    # generate shots for alternative, invalid explanations
    # ensure that invalid explanations do not lead to valid completions
    invalid_fns = []
    invalid_completions = []
    tries_left = len(amb_seqs) * 100 // num_invalid + 1

    while len(invalid_fns) < num_invalid and tries_left > 0:
        invalid_candidates = generate_shot_pool(
            n_shots=num_invalid,
            base_fn=org_fn,
            shot_type=invalid_fn_type,
            ambiguous_sequences=amb_seqs,
        )
        for fn_item in invalid_candidates:
            if fn_item not in all_valid_fns:
                invalid_compl = resolve_fn(fn_item, last_step)
                if invalid_compl not in valid_completions:
                    invalid_completions.append(invalid_compl)
                    invalid_fns.append(fn_item)
                    if len(invalid_fns) == num_invalid:
                        break

        tries_left -= 1

    return invalid_fns, invalid_completions


def _get_valid_alternative_funcs(
    org_func: dict,
    ambiguous_sequences: dict,
    num_valid: int,
    org_seq: Optional[str] = None,
) -> Tuple[str, List[dict]]:

    valid_fns = []
    if org_seq is not None and org_seq not in ambiguous_sequences:
        logger.info(f"Provided sequence '{org_seq}' is not ambgiuous.")
        return org_seq, [org_func]

    for sequence, fns in ambiguous_sequences.items():
        if org_func in fns:
            # sample valid ambiguous functions
            if -1 == num_valid:
                valid_fns = fns  # return all valid
            elif len(fns) >= num_valid:
                valid_fns = random.sample(fns, num_valid)
                while org_func not in valid_fns:
                    # ensure to include the original, consistent func
                    valid_fns = random.sample(fns, num_valid)
            else:
                logger.warning(
                    "Less than {} valid candidates functions, will return all {} available.".format(
                        num_valid, len(fns)
                    )
                )
                valid_fns = random.sample(fns, len(fns))
            break

    if len(valid_fns) == 0:
        raise KeyError(
            f"Could not find function in ambiguous sequences: {str(org_func)}"
        )

    return sequence, valid_fns


def get_data_with_alternatives(
    csv_input_path: str,
    num_valid: int,
    num_invalid: int,
    invalid_fn_type: str,
    skip_non_text_models=True,
):
    """Based on consistent function determined in Q0, generate data samples for Q2
    Each sample consists of:
        - ambiguous sequence
        - original function/explanation (i.e. the one determined in Q0)
        - valid alternative explanations
        - invalid alternative explanations
        - valid completions
        - invalid completions (based on the invalid explanations)

    """
    base_data = parse_function_and_model_from_csv(csv_input_path)

    # filter data if necessary
    if skip_non_text_models:
        base_data = [
            entry
            for entry in base_data
            if isinstance(get_model_from_string(entry["model"]), OpenAITextModels)
        ]

    logger.info(f"No. base functions: {len(base_data)}")

    amb_seqs = find_ambiguous_integer_sequences()

    data = []

    for entry in tqdm(base_data, desc="Generating data for eval with alternatives"):
        model = get_model_from_string(entry["model"])
        consistent_func = entry["fn_item"]
        # {'fn': 'lambda x: (1 * x) ** 1', 'offset': 0, 'metadata': ('exponential_progression', 0, 1)}

        # generate dataset for this eval:
        # 1) generate ambiguous sequence given a valid explanation and find alternative, valid explanation
        # 2) generate valid completions
        # 3) generate shots for invalid explanations
        # 4) generate invalid completions -> cross check that there is no overlap with valid completions

        # find alternative, valid function
        try:
            sequence, valid_fns = _get_valid_alternative_funcs(
                consistent_func, amb_seqs, num_valid
            )
        except KeyError as e:
            logger.error(repr(e))
            continue
        # roll out valid fns to obtain valid completions
        last_step = len(sequence.split(","))
        valid_completions = [resolve_fn(fn_item, last_step) for fn_item in valid_fns]

        invalid_fns = []
        invalid_completions = []

        if 0 < num_invalid:
            invalid_fns, invalid_completions = _generate_invalid_alternatives(
                num_invalid,
                invalid_fn_type,
                consistent_func,
                last_step,
                amb_seqs,
                valid_fns,
                valid_completions,
            )

            if len(invalid_fns) < num_invalid:
                logger.warning(
                    "Not enough invalid candidate functions found. Tried 100 times. Will skip this one."
                )
                continue

        entry = {}
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


def get_data_with_valid_alternatives_only(csv_input_path: str, model: BaseModel):
    # for Q2.2 we only need valid alternatives
    base_data = parse_function_and_model_from_csv(csv_input_path)

    # filter by model
    base_data = [entry for entry in base_data if entry["model"] == model.value]
    logger.info(f"For '{model.value}' found {len(base_data)} consistent base functions")

    amb_seqs = find_ambiguous_integer_sequences()
    data = {}

    for entry in tqdm(base_data, desc="Generating data for Q2.2 eval"):

        consistent_func = entry["fn_item"]
        # {'fn': 'lambda x: (1 * x) ** 1', 'offset': 0, 'metadata': ('exponential_progression', 0, 1)}

        # generate dataset for this eval:
        # 1) generate ambiguous sequence given a valid explanation and find alternative, valid explanation
        # 2) generate valid completions

        # get ALL alternative, valid functions for this sequence
        sequence, valid_fns = _get_valid_alternative_funcs(
            consistent_func, amb_seqs, num_valid=-1
        )

        if sequence not in data:
            # roll out valid fns to obtain valid completions
            last_step = len(sequence.split(","))
            valid_completions = [
                resolve_fn(fn_item, last_step) for fn_item in valid_fns
            ]

            entry = {}
            valids = list(zip(valid_fns, valid_completions))
            random.shuffle(valids)
            valid_fns, valid_completions = zip(*valids)

            entry["valid_fns"] = list(valid_fns)
            entry["valid_completions"] = valid_completions
            entry["model"] = model

            data[sequence] = entry

    return amb_seqs, data
