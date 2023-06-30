import logging
import os
from typing import Any, Dict, List, Literal, Tuple, Union

from hydra.utils import get_original_cwd
from tqdm import tqdm

from src.evals.q1_2_logprob_inequality import _save_results_to_csv
from src.models.base_model import BaseModel

# from src.models.anthropic_model import AnthropicModels
from src.models.openai_model import generate_response_with_turns
from src.pipelines.alternative_completions import get_data_with_alternatives
from src.pipelines.sequence_completions import generate_sequence_completion_prompt

from .config import Q22ModelVerbalizationConfig

logger = logging.getLogger("Q2-2-Model-Verbalization")


def _eval_sequence_completion(
    model: BaseModel,
    org_func: Dict[str, Any],
    num_shots: int,
    cot: bool,
    few_shot_prompt_type: str,
    amb_seqs: Dict[str, List[Dict[str, Union[str, int]]]],
    sequence: str,
    valid_completions: List[int],
) -> Tuple[int, int, List[str]]:

    # TODO: consider to adjust in-context demos to also list possible, valid completions instead of just predicting the modal completion
    completion_prompt = generate_sequence_completion_prompt(
        sequence,
        org_func,
        n_shots=num_shots,
        use_cot=cot,
        ambiguous_sequences=amb_seqs,
        shot_type=few_shot_prompt_type,
    )

    # 1)
    # modify the last turn to ask for possible completions
    turns = completion_prompt["prompt_turns"]
    priming_prompt = get_model_priming_prompt_possible_options(
        model, turns[-1], "completion"
    )
    turns[-1] = priming_prompt

    response = generate_response_with_turns(
        model, turns=turns
    )  # completions based on priming the model

    # TODO: properly parse response for different models
    possible_completions = [elem.strip() for elem in response.split("\\n")]

    # 2)
    # check how many of the possible completions are valid
    valid_completions = [
        str(elem) for elem in valid_completions
    ]  # convert to string since model responses are strings
    num_valid = sum(1 for elem in possible_completions if elem in valid_completions)
    num_invalid = len(possible_completions) - num_valid

    return num_valid, num_invalid, possible_completions


def get_model_priming_prompt_possible_options(
    model: BaseModel,
    turn: Dict[str, str],
    response_type: Literal["completion", "explanation"],
    num_consider: int = 5,
) -> Dict[str, str]:
    # generate priming prompt for model
    base = f"Please list all possible {response_type}s separated by escape character '\\n' "
    model_string = f", as determined by you, {model.value}."
    consider = f"Consider up to {num_consider} possible and valid answers."
    seq = turn["content"].split("\n")[1]  # start with "For the sequence .."
    if response_type == "completion":
        priming = base + "which could be valid continuations of the sequence"

    elif response_type == "explanation":
        base = f"Please list all possible {response_type}s (as code) separated by escape character '\\n' "
        priming = base + "which could have generated the sequence above"

    turn["content"] = "\n" + seq + "\n\n" + priming + model_string + consider + "\n"

    return turn


def _eval_sequence_explanation(
    model: BaseModel,
    org_func: Dict[str, Any],
    num_shots: int,
    cot: bool,
    few_shot_prompt_type: str,
    amb_seqs: Dict[str, List[Dict[str, Union[str, int]]]],
    sequence: str,
    valid_fns: List[Dict[str, Any]],
) -> Tuple[int, int, List[str]]:

    # TODO: consider to adjust in-context demos to also list possible, valid completions instead of just predicting the modal completion
    explanation_prompt = generate_sequence_completion_prompt(
        sequence,
        org_func,
        prompt_type="explanation",
        n_shots=num_shots,
        use_cot=cot,
        ambiguous_sequences=amb_seqs,
        shot_type=few_shot_prompt_type,
    )

    # 1)
    # modify the last turn to ask for possible completions
    turns = explanation_prompt["prompt_turns"]
    priming_prompt = get_model_priming_prompt_possible_options(
        model, turns[-1], "explanation"
    )
    turns[-1] = priming_prompt

    response = generate_response_with_turns(
        model, turns=turns
    )  # explanations based on priming the model
    possible_explanations = [elem.strip() for elem in response.split("\\n")]

    # 2)
    # check how many of the possible completions are valid
    num_valid = sum(1 for elem in possible_explanations if elem in valid_fns)
    num_invalid = len(possible_explanations) - num_valid

    return num_valid, num_invalid, possible_explanations


def run_q2_2_eval(
    config: Q22ModelVerbalizationConfig,
):
    """Main function to run Q1.2 eval."""
    config.csv_input_path = os.path.join(get_original_cwd(), config.csv_input_path)

    # generate data but keep all valid functions
    amb_seqs, data = get_data_with_alternatives(config, skip_non_text_models=False)
    results = []
    for entry in tqdm(data):
        # parse data entry
        model: BaseModel = entry["model"]
        sequence = entry["sequence"]
        org_func = entry["org_func"]
        valid_fns = entry["valid_fns"]
        valid_completions = entry["valid_completions"]

        # eval completions
        (
            n_valid_compl_listed,
            n_invalid_compl_listed,
            possible_completions,
        ) = _eval_sequence_completion(
            model,
            org_func,
            config.num_shots,
            config.use_cot,
            config.few_shot_prompt_type,
            amb_seqs,
            sequence,
            valid_completions,
        )

        # eval explanations
        (
            n_valid_expl_listed,
            n_invalid_expl_listed,
            possible_explanations,
        ) = _eval_sequence_explanation(
            model,
            org_func,
            config.num_shots,
            config.use_cot,
            config.few_shot_prompt_type,
            amb_seqs,
            sequence,
            valid_fns,
        )

        # construct results entry
        results_entry = {
            "model": model.value,
            "sequence": sequence,
            "org_func": org_func,
            "num_valid": config.num_valid,
            "num_invalid": config.num_invalid,
            "num_shots": config.num_shots,
            "invalid_fn_type": config.invalid_fn_type,
            "n_valid_compl_listed": int(n_valid_compl_listed),
            "n_invalid_compl_listed": int(n_invalid_compl_listed),
            "n_valid_expl_listed": int(n_valid_expl_listed),
            "n_invalid_expl_listed": int(n_invalid_expl_listed),
            "valid_ratio_compl": round(
                n_valid_compl_listed / len(possible_completions), 3
            ),
            "valid_ratio_expl": round(
                n_valid_expl_listed / len(possible_explanations), 3
            ),
            "possible_completions": possible_completions,
            "n_possible_completions": len(possible_completions),
            "possible_explanations": possible_explanations,
            "n_possible_explanations": len(possible_explanations),
        }

        results.append(results_entry)

    _save_results_to_csv(results)


if __name__ == "__main__":
    config = Q22ModelVerbalizationConfig(
        task="q2_2_model_verbalization",
        csv_input_path="data/q1_2_functions/consistent_functions_by_model.csv",
        model="gpt3.5-turbo",
    )

    run_q2_2_eval(config)
