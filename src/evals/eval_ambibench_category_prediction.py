import argparse
import datetime
import logging
import os
import random
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
from src.structures.ambibench import AmbiBenchDataset

logger = logging.getLogger("EvalAmbiBenchCompletions")


ALL_CATEGORIES_PROMPT = "Possible options for [category withheld] are the following. Please select one of these labels when prompted:\n"
CATEGORY_PREDICTION_PROMPT = (
    "What is your best guess for the [category withheld] above?\nCategory:"
)


def _get_prompt_for_all_categories(categories: List[str], order_type="shuffle") -> str:
    #

    if "shuffle" == order_type:
        random.shuffle(categories)
    if "sort_alpha" == order_type:
        categories = sorted(categories)

    prompt = ALL_CATEGORIES_PROMPT + ", ".join(categories).strip()[:-1] + "\n"
    # for cat in categories:
    #     prompt += f"- {cat}\n"
    return prompt


def _get_prompt_for_multi_choice_categories(categories_string: str) -> str:
    return ALL_CATEGORIES_PROMPT + categories_string


def format_prompt_for_prediction_for_model_type(
    dataset: AmbiBenchDataset, model: str, use_multiple_choice: bool
) -> Tuple[List[dict], List[str]]:

    formatted_prompts: List[dict] = []
    expected_category: List[str] = []

    # construct the prompt, which includes
    # Instruction
    # Set of examples with their completion
    # Possible categories: either multiple choice or all valid categories
    # Prompt for category prediction
    for ex in dataset.examples:
        example_set = ex["prompt"] + ex["completion"] + "\n"

        if use_multiple_choice and dataset.config.n_multiple_choices > 0:
            possible_cats = _get_prompt_for_multi_choice_categories(
                ex["multiple_choice_category"]
            )
        else:

            possible_cats = _get_prompt_for_all_categories(dataset.candidate_categories)

        category_pred = dataset.assistance_prompts["category_prediction"]
        prompt = example_set + possible_cats + category_pred

        if isinstance(model, OpenAIChatModels):
            template = CHAT_PROMPT_TEMPLATE.copy()
            template["content"] = prompt
            formatted_prompts.append(template)

        elif isinstance(model, OpenAITextModels):
            formatted_prompts.append(prompt)

        else:
            raise NotImplementedError(f"No formatting method for '{model}'")

        expected_category.append(ex["salient_category"].strip())

    return formatted_prompts, expected_category


def eval_category_predictions(
    expected_completions: List[str], pred_completions: List[str]
) -> int:
    # compare predictions to ground truth
    correct_predictions = 0
    for (y, pred) in zip(expected_completions, pred_completions):
        # prediction may contain enumeration
        if pred.split(" ")[-1] in [y, y.lower(), y.upper()]:
            correct_predictions += 1
        else:
            logger.debug(f"'{pred}' != '{y}'")

    return correct_predictions


def get_chat_cat_prediction(prompt: Dict[str, str], model: OpenAIChatModels) -> str:
    chat_response = generate_chat_completion([prompt], model=model)
    # parse predicted completion from response, i.e. last char of the last line
    return chat_response.strip()


def get_text_cat_prediction(prompt: str, model: OpenAITextModels) -> str:
    text_response = generate_text_completion(prompt, model=model)
    # parse predicted completion from response, i.e. last char of the last line
    return text_response.strip()


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
        default="ambibench_category_prediction.tsv",
        help="TSV file name where results are stored.",
    )

    parser.add_argument(
        "--use_multiple_choice",
        action="store_true",
        help="Use multiple choice for category prediction if available. If not set will use all categories by default.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # set arguments
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

    # get data
    dataset = load_ambibench_dataset(args.ambibench_data_path)
    logger.info(f"Dataset config: {repr(dataset.config)}")

    # determine whether to use multiple choice or all options
    if args.use_multiple_choice and dataset.config.n_multiple_choices < 1:
        logger.warning(
            "No multiple choice options in the dataset, will use default of all possible categories."
        )
        use_multiple_choice = False
    elif args.use_multiple_choice and dataset.config.n_multiple_choices > 1:
        use_multiple_choice = True
    else:
        use_multiple_choice = False

    (
        formatted_prompts,
        expected_categories,
    ) = format_prompt_for_prediction_for_model_type(dataset, model, use_multiple_choice)
    logger.info(
        f"No. prompts for AmbiBench category prediction: {len(formatted_prompts)}"
    )

    logger.info(f"Start model inference for: {model.value}")
    pred_categories: List[str] = []
    for prompt in tqdm(formatted_prompts):

        if isinstance(model, OpenAIChatModels):
            prediction = get_chat_cat_prediction(prompt, model)
        if isinstance(model, OpenAITextModels):
            prediction = get_text_cat_prediction(prompt, model)

        pred_categories.append(prediction)

    num_correct_predictions = eval_category_predictions(
        expected_categories, pred_categories
    )

    # store results in TSV
    results = {
        "dataset": str(args.ambibench_data_path).split("/")[-1],
        "model": model.value,
        "num_shots": dataset.config.n_shots,
        "multiple_choice": int(use_multiple_choice),
        "num_examples": len(formatted_prompts),
        "num_correct": num_correct_predictions,
        "acc": round(num_correct_predictions / len(expected_categories), 3),
    }
    logger.info(f"Results: {repr(results)}")
    df = pd.DataFrame.from_dict([results])

    if os.path.exists(output_tsv):
        # append
        df = pd.concat([pd.read_csv(output_tsv, sep="\t"), df], ignore_index=True)

    os.makedirs(os.path.dirname(output_tsv), exist_ok=True)
    df.to_csv(output_tsv, sep="\t", index=False, header=True)
