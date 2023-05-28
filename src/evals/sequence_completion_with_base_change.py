import json
import logging

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
                    (
                        correct_consistent_explanations,
                        correct_inconsistent_explanations,
                        incorrect_consistent_explanations,
                        incorrect_inconsistent_explanations,
                        invalid_explanations,
                    ) = self_consistency_evaluation(
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
                    logger.info("oopies")
                    logger.info(f"Error is: {str(e)}")
                else:
                    if sequence in results:
                        results[sequence][
                            "correct_consistent"
                        ] += correct_consistent_explanations
                        results[sequence][
                            "correct_inconsistent"
                        ] += correct_inconsistent_explanations
                        results[sequence][
                            "incorrect_consistent"
                        ] += incorrect_consistent_explanations
                        results[sequence][
                            "incorrect_inconsistent"
                        ] += incorrect_inconsistent_explanations
                        results[sequence]["invalid"] += invalid_explanations
                    else:
                        results[sequence] = {
                            "correct_consistent": correct_consistent_explanations,
                            "correct_inconsistent": correct_inconsistent_explanations,
                            "incorrect_consistent": incorrect_consistent_explanations,
                            "incorrect_inconsistent": incorrect_inconsistent_explanations,
                            "invalid": invalid_explanations,
                        }
                    break
        else:
            pass
            # TODO: have support for general base sequences here

    logger.info("Total is: {str(total)}")

    # Reformat results
    results = reformat_self_consistency_results(results)

    # Save the results
    with open("results.json", "w") as f:
        json.dump(results, f)

    logger.info("Results saved to results.json")
