import logging
import traceback
from dataclasses import asdict

import pandas as pd
from tqdm import tqdm

from src.evals.check_self_consistency import self_consistency_evaluation
from src.evals.config import SequenceCompletionBaseChangeConfig
from src.pipelines.sequence_completions import find_ambiguous_integer_sequences

logger = logging.getLogger(__name__)


def evaluate_compute_dependence_with_base_changes(
    config: SequenceCompletionBaseChangeConfig,
) -> None:
    total = 0
    seed = config.seed
    results = {}
    all_data = []

    logger.info("Start evaluating compute dependence with base changes.")
    if config.on_ambiguous_sequences:
        # Get the ambiguous sequences
        # Use default parameters for now
        if config.custom_sequences:
            # Use the custom arguments for the ambiguous sequences
            ambiguous_sequences = find_ambiguous_integer_sequences(
                max_constant_term_one=config.max_constant_term_one,
                max_constant_term_two=config.max_constant_term_two,
                num_steps_to_check=config.num_steps_to_check,
                step_offsets=config.step_offsets,
            )
        else:
            ambiguous_sequences = find_ambiguous_integer_sequences()
        logger.info(f"Found {len(ambiguous_sequences)} ambiguous sequences.")
    else:
        raise NotImplementedError("Not yet implemented for non-ambiguous sequences. ")
        # TODO: have support for general base sequences here
    for sequence in tqdm(ambiguous_sequences):
        # Use a new seed for each sequence
        seed += 1
        # turn the sequence from a string into a list of integers
        int_sequence = [int(x) for x in sequence.split(",")]
        logger.info(f"Total: {total}")
        logger.info(f"Sequence: {sequence}")
        for _ in range(2):
            try:
                all_data += self_consistency_evaluation(
                    model_name=config.model.value,
                    sequence=int_sequence,
                    task_prompt=config.task_prompt,
                    role_prompt=config.role_prompt,
                    base=config.base,
                    shots=config.num_shots,
                    shot_method=config.few_shot_prompt_type,
                    temperature=0.0,
                    samples=config.num_samples,
                    seed=seed,
                    show_function_space=config.show_function_space,
                )
            except Exception as e:
                logger.warning("Error in self consistency evaluation.")
                tb = traceback.format_exc()
                logger.warning(f"An error occurred: {e}\n{tb}")
            else:
                total += config.num_samples
                break

    logger.info(f"Total is: {str(total)}")

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
    correct_consistent_percent = round(correct_consistent / total, 3) * 100
    correct_inconsistent_percent = round(correct_inconsistent / total, 3) * 100
    incorrect_consistent_percent = round(incorrect_consistent / total, 3) * 100
    incorrect_inconsistent_percent = round(incorrect_inconsistent / total, 3) * 100

    # Save the results
    results = [
        {
            **asdict(config),
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
