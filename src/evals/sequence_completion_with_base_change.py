import json
import logging

import numpy as np
import pandas as pd

from src.evals.check_self_consistency import self_consistency_evaluation
from src.pipelines.sequence_completions import (
    find_ambiguous_integer_sequences,
    sequence_functions,
)
from src.utils import auto_subdir, reformat_self_consistency_results

logger = logging.getLogger(__name__)


@auto_subdir
def evaluate_compute_dependence_with_base_changes(
    sequence_type: str,
    model: str,
    num_shots: int,
    on_ambiguous_sequences: bool,
    num_samples: int,
    distribution: str = "default",
    shot_method: str = "random",
):
    total = 0
    if on_ambiguous_sequences:
        if sequence_type == "integer":
            base = 10
        elif sequence_type == "binary":
            base = 2
        else:
            raise ValueError("Unknown sequence type.")
        # Get the ambiguous sequences
        # Use default parameters for now
        results = {}
        all_data = []
        ambiguous_sequences = find_ambiguous_integer_sequences(
            valid_sequence_functions={
                fn: v
                for fn, v in sequence_functions.items()
                if fn
                != "indexing_criteria_progression"  # does not work with base change
            },
        )
        for sequence in ambiguous_sequences:
            # turn the sequence from a string into a list of integers
            int_sequence = [int(x) for x in sequence.split(",")]
            total += 1
            logger.info(f"Total: {total}")
            logger.info(f"Sequence: {sequence}")
            for _ in range(2):
                try:
                    logger.info(f"base be: {base}")
                    all_data += self_consistency_evaluation(
                        model_name=model,
                        sequence=int_sequence,
                        distribution=distribution,
                        base=base,
                        shots=num_shots,
                        shot_method=shot_method,
                        temperature=0.0,
                        samples=num_samples,
                    )
                except Exception as e:
                    logger.warning("Error in self consistency evaluation.")
                    logger.error(f"Error is: {str(e)}")
                else:
                    break
        else:
            pass
            # TODO: have support for general base sequences here

    logger.info("Total is: {str(total)}")

    # Log total data
    pd.DataFrame(all_data).to_csv(f"all_data.csv")

    (
        correct_consistent,
        correct_inconsistent,
        incorrect_consistent,
        incorrect_inconsistent,
        invalid,
    ) = (
        [],
        [],
        [],
        [],
        [],
    )
    for data in all_data:
        correct_consistent.append(1 if data["correct"] and data["consistent"] else 0)
        correct_inconsistent.append(
            1 if data["correct"] and not data["consistent"] else 0
        )
        incorrect_consistent.append(
            1 if not data["correct"] and data["consistent"] else 0
        )
        incorrect_inconsistent.append(
            1 if not data["correct"] and not data["consistent"] else 0
        )
        invalid = 1 if data["invalid"] else 0

    correct_consistent_percent = round(np.mean(correct_consistent), 2) * 100
    correct_inconsistent_percent = round(np.mean(correct_inconsistent), 2) * 100
    incorrect_consistent_percent = round(np.mean(incorrect_consistent), 2) * 100
    incorrect_inconsistent_percent = round(np.mean(incorrect_inconsistent), 2) * 100
    invalid_percent = round(np.mean(invalid), 2) * 100

    # Save the results
    results = [
        {
            "model": model,
            "sequence_type": sequence_type,
            "total": total,
            "correct_consistent": correct_consistent_percent,
            "correct_inconsistent": correct_inconsistent_percent,
            "incorrect_consistent": incorrect_consistent_percent,
            "incorrect_inconsistent": incorrect_inconsistent_percent,
            "invalid": invalid_percent,
        }
    ]
    pd.DataFrame(results).to_csv(f"results.csv")

    logger.info("Results saved to results.csv")
