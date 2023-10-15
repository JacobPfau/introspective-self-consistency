import logging

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.evals.config import SequenceCompletionCapabilityConfig
from src.models import BaseModel
from src.models.completions import generate_response_with_turns
from src.pipelines import ShotSamplingType, TaskType
from src.pipelines.sequence_completions import generate_sequence_completion_prompt

logger = logging.getLogger(__name__)


def _sequence_completion_eval(
    sequence: str,
    fn: str,
    model: BaseModel,
    max_offset: int = 8,
    num_shots: int = 8,
    ambiguous_sequences: dict = None,
    few_shot_prompt_type: ShotSamplingType = ShotSamplingType.RANDOM,
    last_sequence_item: int = None,
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

    return {
        "original_function": fn,
        "sequence": sequence,
        "generated_completion_rule": explanation,
        "generated_completion": actual_completion,
        "generated_rule_matches": last_completion_step is not None,
        "generated_completion_matches": str(actual_completion)
        == str(last_sequence_item),
    }


def evaluate_sequence_completion_capability(config: SequenceCompletionCapabilityConfig):
    logger.info("Evaluating sequence completion capability...")
    df = pd.read_csv(config.csv_input_path)
    fns = list(df["fn"])
    total_sequences = len(fns)
    completion_data = []
    for fn in tqdm(fns):
        try:
            sequence_len = np.random.randint(3, 10)
            sequence_raw = [eval(fn)(i) for i in range(sequence_len + 1)]
            sequence = ",".join([str(item) for item in sequence_raw[:-1]])
            last_sequence_item = sequence_raw[-1]
            completion_data.append(
                _sequence_completion_eval(
                    sequence=sequence,
                    fn={"fn": fn, "offset": 0},
                    model=config.model,
                    max_offset=config.max_offset,
                    num_shots=config.num_shots,
                    ambiguous_sequences=None,
                    few_shot_prompt_type=config.few_shot_prompt_type,
                    last_sequence_item=last_sequence_item,
                )
            )
        except Exception as e:
            logger.exception(e)
            logger.warning(e)

    pd.DataFrame(completion_data).to_csv(
        f"sequence_completion_capability_evaluation_{config.model.value}.csv",
        index=False,
    )

    rule_accs, completion_accs = [], []
    for data in completion_data:
        rule_accs.append(1 if data["generated_rule_matches"] else 0)
        completion_accs.append(1 if data["generated_completion_matches"] else 0)

    rule_matches_sequence = round(np.mean(rule_accs), 2) * 100
    completion_is_correct = round(np.mean(completion_accs), 2) * 100
    logger.info(
        f"""
        Evaluated {len(completion_data)} ambiguous sequences of {total_sequences} total.
        Resulting in:
        - {rule_matches_sequence}% rules_matches_sequence
        - {completion_is_correct}% completion_is_correct
        """
    )
