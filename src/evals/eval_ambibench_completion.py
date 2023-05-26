"""This module provides methods to evaluate the correctness and consistency of predicitons
 on basic AmbiBench-style examples. The model is given a prompt with few-shot examples and tested for the following cases:
    1. uninformative completion: predict a binary output {X; Y} on whether or not the last question contains the [category withheld]

Here 'uninformative' refers to [category withheld] in the instruction, i.e. the model is not informed about the category

We are interested in different metrics of correctness and constistency as follows:
    1. ground-truth correctness: Does the model correctly predict the completion (X or Y) or category (multiple-choice)?
    2. self-rule-following consistency: Is the predicted completion of the model consistent with the predicted category?
    3. self-comparison consistency:

 """
import argparse
import datetime
import logging
import os
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from src.models.openai_model import (
    CHAT_PROMPT_TEMPLATE,
    OpenAIChatModels,
    OpenAITextModels,
    generate_chat_completion,
    generate_text_completion,
)
from src.models.utils import get_model_from_string
from src.pipelines.basic_ambibench_completions import load_ambibench_dataset

logger = logging.getLogger("EvalAmbiBenchCompletions")


def format_completion_prompt_for_model_type(
    examples, model: str
) -> Tuple[List[dict], List[str]]:

    formatted_prompts: List[dict] = []
    expected_completions: List[str] = []

    if isinstance(model, OpenAIChatModels):
        # Need to format the original prompt structure to run inference for AmbiBench completions?
        # E.g.: 'Output 'X' if the sentence contains a [category withheld] and 'Y' otherwise.\nQ: The bear is in the prairie.\nA: Y
        #   \nQ: The fugitive is in the river.\nA: Y\nQ: The surveyor is in the marsh.\nA:'
        template = CHAT_PROMPT_TEMPLATE
        for ex in examples:
            template["content"] = ex["prompt"]
            formatted_prompts.append(template)
            expected_completions.append(ex["completion"].strip())
    elif isinstance(model, OpenAITextModels):
        for ex in examples:
            formatted_prompts.append(ex["prompt"])
            expected_completions.append(ex["completion"].strip())
    else:
        raise NotImplementedError(f"No formatting method for '{model}'")

    return formatted_prompts, expected_completions


def eval_completions(
    expected_completions: List[str], pred_completions: List[str]
) -> int:
    # compare predictions to ground truth
    correct_predictions = 0
    for (y, pred) in zip(expected_completions, pred_completions):
        if pred in [y, y.lower(), y.upper()]:
            correct_predictions += 1
        else:
            logger.debug(f"'{pred}' != '{y}'")

    return correct_predictions


def get_chat_completion(prompt: Dict[str, str], model: OpenAIChatModels) -> str:
    completion_response = generate_chat_completion([prompt], model=model)
    # parse predicted completion from response, i.e. last char of the last line
    return completion_response.strip()


def get_text_completion(prompt: str, model: OpenAITextModels) -> str:
    completion_response = generate_text_completion(prompt, model=model)
    # parse predicted completion from response, i.e. last char of the last line
    return completion_response.strip()


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ambibench_data_path",
        type=str,
        required=True,
        default="./data/ambi-bench/dataset.json",
        help="File path to dataset",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="gpt-3.5-turbo",
        help="Model name for which to run evals",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="./results",
        help="Directory where to store results",
    )

    parser.add_argument(
        "--output_tsv",
        type=str,
        required=False,
        default="ambibench_completions.tsv",
        help="TSV file name where results are stored.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = _get_args()

    model = get_model_from_string(args.model)

    date = datetime.datetime.now().strftime("%y%m%d")
    if args.output_tsv is not None:
        output_tsv = os.path.join(args.output_dir, f"{date}_" + args.output_tsv)
    else:
        # use dataset name for results file
        tsv_name = (
            str(args.ambibench_data_path)
            .split("/")[-1]
            .replace(".json", "_results.tsv")
        )
        output_tsv = os.path.join(args.output_dir, f"{date}_" + tsv_name)

    ###

    #
    dataset = load_ambibench_dataset(args.ambibench_data_path)
    logger.info(f"Dataset config: {repr(dataset.config)}")

    formatted_prompts, expected_completions = format_completion_prompt_for_model_type(
        dataset.examples, model
    )
    logger.info(f"No. prompts for AmbiBench completion: {len(formatted_prompts)}")

    logger.info(f"Start model inference for: {model.value}")
    pred_completions: List[str] = []
    for prompt in tqdm(formatted_prompts):
        if isinstance(model, OpenAIChatModels):
            completion = get_chat_completion(prompt, model)
        if isinstance(model, OpenAITextModels):
            completion = get_text_completion(prompt, model)

        pred_completions.append(completion)

    num_correct_predictions = eval_completions(expected_completions, pred_completions)

    # store results in TSV
    results = {
        "dataset": str(args.ambibench_data_path).split("/")[-1],
        "model": model.value,
        "num_shots": dataset.config.n_shots,
        "num_examples": len(formatted_prompts),
        "num_correct": num_correct_predictions,
        "acc": round(num_correct_predictions / len(expected_completions), 3),
    }
    logger.info(f"Results: {repr(results)}")
    df = pd.DataFrame.from_dict(results, orient="index")

    if os.path.exists(output_tsv):
        # append
        df = pd.concat([pd.read_csv(output_tsv, sep="\t"), df], ignore_index=True)

    os.makedirs(os.path.dirname(output_tsv), exist_ok=True)
    df.to_csv(output_tsv, sep="\t", index=False, header=True)
