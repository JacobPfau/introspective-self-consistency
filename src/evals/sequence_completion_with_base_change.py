import logging

import pandas as pd

from src.evals.check_self_consistency import self_consistency_evaluation
from src.evals.config import SequenceCompletionBaseChangeConfig
from src.pipelines.sequence_completions import find_ambiguous_integer_sequences

logger = logging.getLogger(__name__)


def evaluate_compute_dependence_with_base_changes(
    config: SequenceCompletionBaseChangeConfig,
) -> None:
    total = 0

    if config.on_ambiguous_sequences:
        if config.sequence_type == "integer":
            base = 10
        elif config.sequence_type == "binary":
            base = 2
        else:
            raise ValueError("Unknown sequence type.")
        # Get the ambiguous sequences
        # Use default parameters for now
        results = {}
        all_data = []
        ambiguous_sequences = find_ambiguous_integer_sequences()
        for sequence in ambiguous_sequences:
            # turn the sequence from a string into a list of integers
            int_sequence = [int(x) for x in sequence.split(",")]
            logger.info(f"Total: {total}")
            logger.info(f"Sequence: {sequence}")
            for _ in range(2):
                try:
                    logger.info(f"base be: {base}")
                    all_data += self_consistency_evaluation(
                        model_name=config.model.value,
                        sequence=int_sequence,
                        distribution=config.distribution,
                        base=base,
                        shots=config.num_shots,
                        shot_method=config.shot_method,
                        temperature=0.0,
                        samples=config.num_samples,
                    )
                except Exception as e:
                    logger.warning("Error in self consistency evaluation.")
                    logger.error(str(e))
                else:
                    total += 1
                    break
        else:
            pass
            # TODO: have support for general base sequences here

    logger.info("Total is: {str(total)}")

    # Log total data
    pd.DataFrame(all_data).to_csv("all_data.csv")

    (
        correct_consistent,
        correct_inconsistent,
        incorrect_consistent,
        incorrect_inconsistent,
        invalid,
    ) = (
        0,
        0,
        0,
        0,
        0,
    )
    for data in all_data:
        if data["invalid"]:
            invalid += 1
            continue

        correct_consistent += 1 if data["correct"] and data["consistent"] else 0
        correct_inconsistent += 1 if (data["correct"] and not data["consistent"]) else 0
        incorrect_consistent += 1 if (not data["correct"] and data["consistent"]) else 0
        incorrect_inconsistent += (
            1 if (not data["correct"] and not data["consistent"]) else 0
        )
    correct_consistent_percent = round(correct_consistent / total, 2) * 100
    correct_inconsistent_percent = round(correct_inconsistent / total, 2) * 100
    incorrect_consistent_percent = round(incorrect_consistent / total, 2) * 100
    incorrect_inconsistent_percent = round(incorrect_inconsistent / total, 2) * 100

    # Save the results
    results = [
        {
            "model": config.model,
            "sequence_type": config.sequence_type,
            "total sequences": total,
            "invalid sequences": invalid,
            "valid sequences": total - invalid,
            "correct_consistent": correct_consistent_percent,
            "correct_inconsistent": correct_inconsistent_percent,
            "incorrect_consistent": incorrect_consistent_percent,
            "incorrect_inconsistent": incorrect_inconsistent_percent,
        }
    ]
    pd.DataFrame(results).to_csv("results.csv")

    logger.info("Results saved to results.csv")
