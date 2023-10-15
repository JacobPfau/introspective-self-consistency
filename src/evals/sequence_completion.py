import logging

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.evals.config import SequenceCompletionEqConfig
from src.models.completions import generate_response_with_turns
from src.pipelines import ShotSamplingType, TaskType
from src.pipelines.consistency_check import (
    generate_completion_check_prompt,
    generate_consistency_check_prompt,
)
from src.pipelines.sequence_completions import (
    find_ambiguous_integer_sequences,
    generate_sequence_completion_prompt,
)

logger = logging.getLogger(__name__)


def _sequence_completion_equality(
    sequence: str,
    fn: str,
    model: str,
    max_offset: int = 8,
    num_shots: int = 8,
    evaluate_model_completion: bool = True,
    evaluate_model_consistency: bool = True,
    ambiguous_sequences: dict = None,
    few_shot_prompt_type: ShotSamplingType = ShotSamplingType.RANDOM,
):
    completion_prompt = generate_sequence_completion_prompt(
        sequence=sequence,
        fn_item=fn,
        n_shots=num_shots,
        ambiguous_sequences=ambiguous_sequences,
        shot_type=few_shot_prompt_type,
    )

    explanation_prompt = generate_sequence_completion_prompt(
        sequence=sequence,
        fn_item=fn,
        n_shots=num_shots,
        task_type=TaskType.EXPLANATION,
        ambiguous_sequences=ambiguous_sequences,
        shot_type=few_shot_prompt_type,
    )

    completion_resp = generate_response_with_turns(
        model, completion_prompt["prompt_turns"]
    )
    explanation_resp = generate_response_with_turns(
        model, explanation_prompt["prompt_turns"]
    )
    explanation = [strs for strs in explanation_resp.split("\n") if strs.strip()][
        0
    ].strip()
    actual_completion = [strs for strs in completion_resp.split("\n") if strs.strip()][
        0
    ].strip()

    # use the model to check if the completion is consistent with the explanation
    consistency_resp = None
    if evaluate_model_consistency:
        consistency_prompt = generate_consistency_check_prompt(
            sequence + f",{actual_completion}", explanation
        )
        consistency_resp_raw = generate_response_with_turns(
            model, [{"role": "user", "content": consistency_prompt}]
        )
        consistency_resp = consistency_resp_raw.split("\n")[0].strip()

    # check what completion the model would have generated given the explanation/rule
    model_completion_resp = None
    if evaluate_model_completion:
        model_completion_prompt = generate_completion_check_prompt(
            sequence, explanation
        )
        model_completion_resp_raw = generate_response_with_turns(
            model, [{"role": "user", "content": model_completion_prompt}]
        )
        model_completion_resp = model_completion_resp_raw.split("\n")[0].strip()

    # find the offset that generates the sequence
    sequence = [int(item) for item in sequence.split(",")]
    last_completion_step = None
    sequence_matched = []
    for i in range(max_offset):
        completion = eval(explanation)(i)
        if completion in sequence:
            sequence_matched.append(completion)
        if sequence_matched == sequence:
            last_completion_step = i
            break

    if last_completion_step is None:
        return {
            "original_function": fn,
            "sequence": sequence,
            "generated_completion_rule": explanation,
            "generated_completion": actual_completion,
            "generated_completion_matches": False,
            "model_self_consistency_evaluation": consistency_resp,
            "model_completion": model_completion_resp,
            "model_completion_matches": False,
        }

    last_completion = eval(explanation)(last_completion_step + 1)
    return {
        "original_function": fn,
        "sequence": sequence,
        "generated_completion_rule": explanation,
        "generated_completion": actual_completion,
        "generated_completion_matches": int(actual_completion) == last_completion,
        "model_self_consistency_evaluation": consistency_resp,
        "model_completion": model_completion_resp,
        "model_completion_matches": int(model_completion_resp) == last_completion,
    }


def evaluate_sequence_completion_equality(config: SequenceCompletionEqConfig) -> None:
    max_offset = config.max_offset
    num_shots = config.num_shots
    few_shot_prompt_type = ShotSamplingType(config.few_shot_prompt_type)

    logger.info("Evaluating sequence completion equality...")
    ambiguous_sequences = find_ambiguous_integer_sequences()
    total_sequences = sum(len(fns) for fns in ambiguous_sequences.values())
    completion_data = []
    for sequence, fns in tqdm(list(ambiguous_sequences.items())):
        for fn in fns:
            try:
                completion_data.append(
                    _sequence_completion_equality(
                        sequence=sequence,
                        fn=fn,
                        model=config.model,
                        max_offset=max_offset,
                        num_shots=num_shots,
                        ambiguous_sequences=ambiguous_sequences,
                        few_shot_prompt_type=few_shot_prompt_type,
                    )
                )
            except Exception as e:
                logger.exception(e)
                logger.warning(e)

    pd.DataFrame(completion_data).to_csv(
        f"sequence_completion_equality_evaluation_{config.model.value}.csv", index=False
    )

    (
        match_accs,
        model_match_accs,
        model_consistency_accs,
        consistent_and_matched_positive,
        consistent_and_matched_negative,
    ) = ([], [], [], [], [])
    for data in completion_data:
        match_accs.append(1 if data["generated_completion_matches"] else 0)
        model_match_accs.append(1 if data["model_completion_matches"] else 0)
        model_consistency_accs.append(
            1 if data["model_self_consistency_evaluation"].strip() == "Y" else 0
        )
        consistent_and_matched_positive.append(
            1
            if data["model_self_consistency_evaluation"].strip() == "Y"
            and data["generated_completion_matches"]
            else 0
        )
        consistent_and_matched_negative.append(
            1
            if data["model_self_consistency_evaluation"].strip() == "N"
            and not data["generated_completion_matches"]
            else 0
        )

    ground_truth_consistent = round(np.mean(match_accs), 2) * 100
    self_rule_following_consistency = round(np.mean(model_match_accs), 2) * 100
    self_comparison_consistency = round(np.mean(model_consistency_accs), 2) * 100
    consistent_and_matched_positive_accuracy = (
        round(np.mean(consistent_and_matched_positive), 2) * 100
    )
    consistent_and_matched_negative_accuracy = (
        round(np.mean(consistent_and_matched_negative), 2) * 100
    )
    logger.info(
        f"""
        Evaluated {len(completion_data)} ambiguous sequences of {total_sequences} total.
        Resulting in:
        - {ground_truth_consistent}% ground-truth-consistent
        - {self_rule_following_consistency}% self-rule-following-consistency
        - {self_comparison_consistency}% self-comparison-consistency
        - {consistent_and_matched_positive_accuracy}% self-comparison-consistency and ground-truth-consistent (positive).
        - {consistent_and_matched_negative_accuracy}% self-comparison-consistency and ground-truth-consistent (negative).
        """
    )
