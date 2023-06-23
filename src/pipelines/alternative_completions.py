import ast
import csv
import logging
import random
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from src.evals.config import Q12LogprobInequalityConfig
from src.models.openai_model import OpenAITextModels
from src.models.utils import get_model_from_string
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


def generate_invalid_alternatives(
    config, org_fn, last_step: int, amb_seqs, all_valid_fns, valid_completions
) -> Tuple[List[dict], List[int]]:
    # generate shots for alternative, invalid explanations
    # ensure that invalid explanations do not lead to valid completions
    invalid_fns = []
    invalid_completions = []
    tries = 0
    while len(invalid_fns) < config.num_invalid and tries < 100:
        invalid_candidates = generate_shot_pool(
            n_shots=config.num_invalid,
            base_fn=org_fn,
            shot_type=config.invalid_fn_type,
            ambiguous_sequences=amb_seqs,
        )
        for fn_item in invalid_candidates:
            if fn_item not in all_valid_fns:
                invalid_compl = resolve_fn(fn_item, last_step)
                if invalid_compl not in valid_completions:
                    invalid_completions.append(invalid_compl)
                    invalid_fns.append(fn_item)

        tries += 1

    return invalid_fns, invalid_completions


def get_data_with_alternatives(
    config: Q12LogprobInequalityConfig, skip_non_text_models=True
):
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

                invalid_fns, invalid_completions = generate_invalid_alternatives(
                    config,
                    consistent_func,
                    last_step,
                    amb_seqs,
                    valid_fns,
                    valid_completions,
                )

                if len(invalid_fns) < config.num_invalid:
                    logger.warning(
                        "Not enough invalid candidate functions found. Tried 100 times. Will skip this one."
                    )
                    continue

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
