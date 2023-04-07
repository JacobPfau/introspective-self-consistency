"""This module provides methods to evaluate the correctness and consistency of predicitons
 on basic AmbiBench-style examples. The model is given a prompt with few-shot examples and tested for the following cases:
    1. uninformative completion: predict a binary output {X; Y} on whether or not the last question contains the [category withheld]
    2. uninformative multiple-choice category prediction: predict which category out of multiple choices explains the examples

Here 'uninformative' refers to [category withheld] in the instruction, i.e. the model is not informed about the category

We are interested in different metrics of correctness and constistency as follows:
    1. ground-truth correctness: Does the model correctly predict the completion (X or Y) or category (multiple-choice)?
    2. self-rule-following consistency: Is the predicted completion of the model consistent with the predicted category?
    3. self-comparison consistency:

 """
import datetime
import logging
import os
from typing import List

import pandas as pd

from src.models.openai_model import CHAT_MODEL_NAME, generate_chat_completion
from src.pipelines.basic_ambibench_completions import load_ambibench_dataset

# from src.structures.ambibench import AmbiBenchDataset

logger = logging.getLogger("EvalAmbiBenchCompletions")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
    level=logging.DEBUG,
)


def format_prompt_for_model_type(prompts, model: str) -> List[dict]:

    formatted_prompts: List[dict] = []

    if CHAT_MODEL_NAME == model:
        # format the original prompt structure to run inference for AmbiBench completions
        pass
    else:
        raise NotImplementedError(f"No formatting method for '{model}'")

    return formatted_prompts


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


if __name__ == "__main__":

    # set params
    ambibench_data_dir = "./data/ambi-bench"
    data_file_name = "20230406_12-28_ambibench_examples.json"
    date = datetime.datetime.now().strftime("%Y%M%D_%H%m")
    output_csv = f"./results/{date}_ambibench_completions.csv"
    model = CHAT_MODEL_NAME

    ###

    #
    dataset = load_ambibench_dataset(os.path.join(ambibench_data_dir, data_file_name))
    logger.info(f"Dataset config: {repr(dataset.config)}")

    formatted_prompts, expected_completions = format_prompt_for_model_type(
        dataset.examples, model
    )
    logger.info(f"No. prompts for AmbiBench completion: {len(formatted_prompts)}")

    logger.info("Start model inference")
    pred_completions = generate_chat_completion(formatted_prompts, model=model)

    correct_predictions = eval_completions(expected_completions, pred_completions)

    # append to results CSV
    results = {
        "dataset": data_file_name,
        "num_examples": len(formatted_prompts),
        "num_correct": correct_predictions,
        "acc": round(correct_predictions / len(formatted_prompts), 3),
    }
    df = pd.DataFrame.from_dict(results)

    if os.path.exists(output_csv):
        # append
        df = pd.concat([pd.read_csv(output_csv), df], ignore_index=True)

    df.to_csv(output_csv, sep=",", index=False, header=True)
