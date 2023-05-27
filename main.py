import logging
from typing import Callable

import hydra
from omegaconf import DictConfig

from src.evals.eval_ambibench_category_prediction import (
    evaluate_ambibench_category_prediction,
)
from src.evals.eval_ambibench_completion import evaluate_ambibench_completion
from src.evals.sequence_completion import evaluate_sequence_completion_equality
from src.evals.sequence_completion_with_base_change import (
    evaluate_compute_dependence_with_base_changes,
)
from src.evals.string_transformation import evaluate_string_transformation_equality
from src.utils import log_exceptions

logger = logging.getLogger(__name__)

TASK_FUNS = {
    "string_transformation_completion_equality": evaluate_string_transformation_equality,
    "sequence_completion_equality": evaluate_sequence_completion_equality,
    "compute_dependence_with_base_changes": evaluate_compute_dependence_with_base_changes,
    "ambibench_category_prediction": evaluate_ambibench_category_prediction,
    "ambibench_completion": evaluate_ambibench_completion,
}


@hydra.main(version_base=None, config_path="conf", config_name="main")
@log_exceptions(logger)
def main(cfg: DictConfig) -> None:
    if not cfg.keys() == {"task", "config"}:
        raise ValueError(
            f"Config should have exactly two keys: 'task' and 'config', but got {cfg.keys()}."
        )
    task: str = cfg.task
    task_cfg: DictConfig = cfg.config
    task_fun: Callable[[DictConfig], None] = TASK_FUNS.get(task)
    if task_fun is None:
        raise ValueError(f"Task '{task}' not supported.")
    task_fun(task_cfg)


if __name__ == "__main__":
    main()
