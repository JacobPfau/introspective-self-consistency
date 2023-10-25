import logging
from typing import Any, Dict, List, Tuple, Union

from tqdm import tqdm

from src.evals.config import Q22ModelVerbalizationConfig
from src.evals.q2_1_logprob_inequality import _save_results_to_csv
from src.models import BaseModel, generate_response_with_turns
from src.pipelines import TaskType
from src.pipelines.alternative_completions import get_data_with_valid_alternatives_only
from src.pipelines.q2_sequence_completions import (
    generate_sequence_completion_prompt_with_valid_continuations,
)

logger = logging.getLogger("Q2-2-Model-Verbalization")


def _eval_sequence_completion(
    sequence: str,
    valid_fns: List[dict],
    valid_completions: List[int],
    num_shots: int,
    max_considerations: int,
    amb_seqs: Dict[str, List[Dict[str, Union[str, int]]]],
    model: BaseModel,
) -> Tuple[int, int, List[str]]:

    completion_prompt = generate_sequence_completion_prompt_with_valid_continuations(
        sequence,
        valid_fns,
        ambiguous_sequences=amb_seqs,
        n_shots=num_shots,
        max_considerations=max_considerations,
        model=model,
    )

    # 1)
    # prompt for possible completions
    response = generate_response_with_turns(
        model, turns=completion_prompt["prompt_turns"]
    )  # completions based on priming the model

    possible_completions = [elem.strip() for elem in response.split("\\n")]
    if len(possible_completions) > max_considerations:
        logger.warning(
            "Model returned {} completions, \
                       which is more than max of {}.".format(
                len(possible_completions), max_considerations
            )
        )
        possible_completions = possible_completions[:max_considerations]

    # 2)
    # check how many of the possible completions are valid
    # determine precision and recall
    # only use distinct valid completions
    valid_completions = [
        str(elem) for elem in set(valid_completions)
    ]  # convert to string since model responses are strings

    tp = sum(
        1 for elem in possible_completions if elem in valid_completions
    )  # true positive
    fp = len(possible_completions) - tp  # false positive
    precision = tp / (tp + fp)
    fn = sum(
        1 for elem in valid_completions if elem not in possible_completions
    )  # false negative
    recall = tp / min((tp + fn), max_considerations)

    return tp, fp, fn, precision, recall, possible_completions


def _eval_sequence_explanation(
    sequence: str,
    valid_fns: List[Dict[str, Any]],
    num_shots: int,
    max_considerations: int,
    amb_seqs: Dict[str, List[Dict[str, Union[str, int]]]],
    model: BaseModel,
) -> Tuple[int, int, List[str]]:

    explanation_prompt = generate_sequence_completion_prompt_with_valid_continuations(
        sequence,
        valid_fns,
        ambiguous_sequences=amb_seqs,
        task_type=TaskType.EXPLANATION,
        n_shots=num_shots,
        max_considerations=max_considerations,
        model=model,
    )

    # 1)
    # ask for possible explanations
    response = generate_response_with_turns(
        model, turns=explanation_prompt["prompt_turns"]
    )  # explanations based on priming the model
    possible_explanations = [elem.strip() for elem in response.split("\\n")]

    if len(possible_explanations) > max_considerations:
        logger.warning(
            "Model returned {} explanations, \
                       which is more than max of {}.".format(
                len(possible_explanations), max_considerations
            )
        )
        possible_explanations = possible_explanations[:max_considerations]

    # 2)
    # check how many of the possible explanations are valid
    # determine precision and recall
    valid_fns = {fn_item["fn"] for fn_item in valid_fns}  # select only the functions
    tp = sum(1 for elem in possible_explanations if elem in valid_fns)  # true positive
    fp = len(possible_explanations) - tp  # false positive
    precision = tp / (tp + fp)
    fn = sum(
        1 for elem in valid_fns if elem not in possible_explanations
    )  # false negative
    recall = tp / min((tp + fn), max_considerations)  # adjust for max_considerations

    return tp, fp, fn, precision, recall, possible_explanations


def run_q2_2_eval(
    config: Q22ModelVerbalizationConfig,
):
    """Main function to run Q2.2 eval."""

    # generate data but keep all valid functions
    amb_seqs, data = get_data_with_valid_alternatives_only(config.shot_pool_term)

    results = []

    for sequence, entry in tqdm(data.items()):
        # parse data entry
        valid_fns = entry["valid_fns"]
        valid_completions = entry["valid_completions"]

        # eval completions
        (
            tp,
            fp,
            fn,
            precision,
            recall,
            possible_completions,
        ) = _eval_sequence_completion(
            sequence,
            valid_fns,
            valid_completions,
            config.num_shots,
            config.max_considerations,
            amb_seqs,
            config.model,
        )

        # eval explanations
        (
            expl_tp,
            expl_fp,
            expl_fn,
            expl_precision,
            expl_recall,
            possible_explanations,
        ) = _eval_sequence_explanation(
            sequence,
            valid_fns,
            config.num_shots,
            config.max_considerations,
            amb_seqs,
            config.model,
        )

        # construct results entry
        results_entry = {
            "model": config.model.value,
            "sequence": sequence,
            "num_shots": config.num_shots,
            "max_considerations": config.max_considerations,
            "precision_compl": round(precision, 3),
            "recall_compl": round(recall, 3),
            "tp_compl": tp,
            "fp_compl": fp,
            "fn_compl": fn,
            "precision_expl": round(expl_precision, 3),
            "recall_expl": round(expl_recall, 3),
            "tp_expl": expl_tp,
            "fp_expl": expl_fp,
            "fn_expl": expl_fn,
            "possible_completions": possible_completions,
            "n_possible_completions": len(possible_completions),
            "possible_explanations": possible_explanations,
            "n_possible_explanations": len(possible_explanations),
        }

        results.append(results_entry)

    _save_results_to_csv(results)


if __name__ == "__main__":
    config = Q22ModelVerbalizationConfig(
        task="q2_2_alternative_verbalization",
        model="gpt-3.5-turbo-0301",
    )

    run_q2_2_eval(config)
