import logging

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.models.openai_model import generate_response_with_turns
from src.pipelines.string_transformations import (
    find_ambiguous_string_transformations,
    generate_string_transformation_prompt,
)
from src.utils import auto_subdir

logger = logging.getLogger(__name__)

NUM_SHOTS = 8
COT = False

MODEL_CONSISTENCY_SHOTS = [
    {"transformation": "#@#@ => @#@#", "fn": "lambda x: x[::-1]", "answer": "Y"},
    {"transformation": "#@## => ####", "fn": "lambda x: x[1:] + x[:1]", "answer": "N"},
    {
        "transformation": "@#@# => ####",
        "fn": "lambda x: x.replace('@', '#', 2)",
        "answer": "Y",
    },
    {
        "transformation": "@### => #@@@",
        "fn": "lambda x: x.translate(str.maketrans('@' + '#', '#' + '@'))",
        "answer": "Y",
    },
    {
        "transformation": "#@## => @###",
        "fn": "lambda x: x[:1] + '@' + x[1:].replace('@', " ", 1)",
        "answer": "N",
    },
    {
        "transformation": "@### => ###@",
        "fn": "lambda x: x[-3:] + x[:-3]",
        "answer": "Y",
    },
]
MODEL_CONSISTENCY_CHECK_PROMPT = """
Is the following transformation: {transformation} consistent with the function {fn}?
Answer (Y/N):"""

MODEL_COMPLETION_SHOTS = [
    {"transformation": "#@#@ => @#@#", "fn": "lambda x: x[::-1]", "answer": "#@#@"},
    {
        "transformation": "#@## => @###",
        "fn": "lambda x: x[1:] + x[:1]",
        "answer": "###@",
    },
    {
        "transformation": "@#@# => ####",
        "fn": "lambda x: x.replace('@', '#', 2)",
        "answer": "####",
    },
    {
        "transformation": "@### => #@@@",
        "fn": "lambda x: x.translate(str.maketrans('@' + '#', '#' + '@'))",
        "answer": "@###",
    },
    {
        "transformation": "#@## => #@##",
        "fn": "lambda x: x[:1] + '@' + x[1:].replace('@', " ", 1)",
        "answer": "#@##",
    },
    {
        "transformation": "@### => ###@",
        "fn": "lambda x: x[-3:] + x[:-3]",
        "answer": "##@#",
    },
]
MODEL_COMPLETION = """
The {transformation} is generated by the function {fn}
Apply the transformation to:
{transformed_string} =>"""


def _generate_consistency_check_prompt(transformation, fn):
    prompt = ""
    for shot in MODEL_CONSISTENCY_SHOTS:
        prompt += (
            MODEL_CONSISTENCY_CHECK_PROMPT.format(
                transformation=shot["transformation"], fn=shot["fn"]
            )
            + " "
            + shot["answer"]
        )
    prompt += MODEL_CONSISTENCY_CHECK_PROMPT.format(
        transformation=transformation, fn=fn
    )
    return prompt


def _generate_completion_check_prompt(transformation, fn):
    prompt = ""
    for shot in MODEL_COMPLETION_SHOTS:
        prompt += (
            MODEL_COMPLETION.format(
                transformation=shot["transformation"],
                fn=shot["fn"],
                transformed_string=shot["answer"],
            )
            + " "
            + str(shot["answer"])
        )
    prompt += MODEL_COMPLETION.format(
        transformation=transformation,
        fn=fn,
        transformed_string=transformation.split(" => ")[-1].strip(),
    )
    return prompt


def string_transformation_equality(
    sequence: str,
    fn: str,
    model: str,
    sequence_length: int,
    char_1: str,
    char_2: str,
    num_shots=NUM_SHOTS,
    cot=COT,
    evaluate_model_completion=True,
    evaluate_model_consistency=True,
):
    completion_prompt = generate_string_transformation_prompt(
        sequence,
        {"fn": fn},
        sequence_length,
        char_1,
        char_2,
        n_shots=num_shots,
        use_cot=cot,
    )
    explanation_prompt = generate_string_transformation_prompt(
        sequence,
        {"fn": fn},
        sequence_length,
        char_1,
        char_2,
        n_shots=num_shots,
        use_cot=cot,
        prompt_type="explanation",
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
        consistency_prompt = _generate_consistency_check_prompt(sequence, explanation)
        consistency_resp = generate_response_with_turns(
            model, [{"role": "user", "content": consistency_prompt}]
        ).strip()

    # # check what the model would have generated
    model_completion_resp = None
    if evaluate_model_completion:
        model_completion_prompt = _generate_completion_check_prompt(
            sequence, explanation
        )
        model_completion_resp = generate_response_with_turns(
            model, [{"role": "user", "content": model_completion_prompt}]
        ).strip()

    # find the offset that generates the sequence
    last_completion = eval(explanation)(sequence.split(" => ")[-1].strip())
    logger.info("last_completion", last_completion)
    logger.info("actual_completion", actual_completion)
    logger.info("model_completion_resp", model_completion_resp)
    logger.info("consistency_resp", consistency_resp)

    return {
        "original_function": fn,
        "sequence": sequence,
        "generated_completion_rule": explanation,
        "generated_completion": actual_completion,
        "generated_completion_matches": actual_completion == last_completion,
        "model_self_consistency_evaluation": consistency_resp,
        "model_completion": model_completion_resp,
        "model_completion_matches": model_completion_resp == last_completion,
    }


@auto_subdir
def evaluate_string_transformation_equality(model, num_shots=NUM_SHOTS, cot=COT):
    logger.info("Evaluating string transformation equality...")
    ambiguous_sequences = find_ambiguous_string_transformations("#", "@", 4)
    total_sequences = sum(len(fns) for fns in ambiguous_sequences.values())
    completion_data = []
    for sequence, fns in tqdm(list(ambiguous_sequences.items())):
        for fn in fns:
            try:
                completion_data.append(
                    string_transformation_equality(
                        sequence,
                        fn,
                        model,
                        4,
                        "#",
                        "@",
                        num_shots=num_shots,
                        cot=cot,
                    )
                )
            except Exception as e:
                logger.warning(e)
                logger.warning(f"Failed to evaluate {sequence} with {fn}")
                continue

    pd.DataFrame(completion_data).to_csv(f"results_{model}.csv", index=False)

    match_accs, model_match_accs, model_consistency_accs, consistent_and_matched = (
        [],
        [],
        [],
        [],
    )
    for data in completion_data:
        match_accs.append(1 if data["generated_completion_matches"] else 0)
        model_match_accs.append(1 if data["model_completion_matches"] else 0)
        model_consistency_accs.append(
            1 if data["model_self_consistency_evaluation"].strip() == "Y" else 0
        )
        consistent_and_matched.append(
            1
            if data["model_self_consistency_evaluation"].strip() == "Y"
            and data["generated_completion_matches"]
            else 0
        )

    ground_truth_consistent = round(np.mean(match_accs), 2) * 100
    self_rule_following_consistency = round(np.mean(model_match_accs), 2) * 100
    self_comparison_consistency = round(np.mean(model_consistency_accs), 2) * 100
    consistent_and_matched_accuracy = round(np.mean(consistent_and_matched), 2) * 100
    logger.info(
        f"""
        Evaluated {len(completion_data)} ambiguous string transformations of {total_sequences} total.
        Resulting in:
        - {ground_truth_consistent}% ground-truth-consistent
        - {self_rule_following_consistency}% self-rule-following-consistency
        - {self_comparison_consistency}% self-comparison-consistency
        - {consistent_and_matched_accuracy}% self-comparison-consistency and ground-truth-consistent.
        """
    )
