import logging
from typing import Callable, Tuple

import hydra
from omegaconf import DictConfig

from src.evals.config import (
    AmbibenchCatPredConfig,
    AmbibenchCompletionConfig,
    BaseEvalConfig,
    Q12LogprobInequalityConfig,
    SequenceCompletionBaseChangeConfig,
    SequenceCompletionEqConfig,
    StringTransformationConfig,
)
from src.evals.eval_ambibench_category_prediction import (
    evaluate_ambibench_category_prediction,
)
from src.evals.eval_ambibench_completion import evaluate_ambibench_completion
from src.evals.q1_2_logprob_inequality import run_q1_2_eval
from src.evals.sequence_completion import evaluate_sequence_completion_equality
from src.evals.sequence_completion_with_base_change import (
    evaluate_compute_dependence_with_base_changes,
)
from src.evals.string_transformation import evaluate_string_transformation_equality
from src.utils import log_exceptions

logger = logging.getLogger(__name__)

TASK_FUNS = {
    "string_transformation_completion_equality": {
        "fn": evaluate_string_transformation_equality,
        "config": StringTransformationConfig,
    },
    "sequence_completion_equality": {
        "fn": evaluate_sequence_completion_equality,
        "config": SequenceCompletionEqConfig,
    },
    "compute_dependence_with_base_changes": {
        "fn": evaluate_compute_dependence_with_base_changes,
        "config": SequenceCompletionBaseChangeConfig,
    },
    "ambibench_category_prediction": {
        "fn": evaluate_ambibench_category_prediction,
        "config": AmbibenchCatPredConfig,
    },
    "ambibench_completion": {
        "fn": evaluate_ambibench_completion,
        "config": AmbibenchCompletionConfig,
    },
    "q1_2_logprob_inequality": {
        "fn": run_q1_2_eval,
        "config": Q12LogprobInequalityConfig,
    },
}


def get_task_and_config(cfg: DictConfig) -> Tuple[Callable, BaseEvalConfig]:
    task: str = cfg.task
    if task not in TASK_FUNS:
        raise ValueError(f"Task '{task}' not supported.")

    task_fun: Callable = TASK_FUNS[task]["fn"]
    config: BaseEvalConfig = TASK_FUNS[task]["config"]
    task_cfg = config.from_dict(cfg)
    return task_fun, task_cfg


@hydra.main(version_base=None, config_path="conf", config_name="main")
@log_exceptions(logger)
def main(cfg: DictConfig) -> None:
    task_fun, task_cfg = get_task_and_config(cfg)
    task_fun(task_cfg)


if __name__ == "__main__":
    main()
