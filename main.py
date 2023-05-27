import logging
from pathlib import Path
from typing import Callable

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

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


def load_default_task_cfg(task: str) -> DictConfig:
    cfg_path = Path(get_original_cwd()) / f"conf/tasks/{task}.yaml"
    try:
        cfg: DictConfig = OmegaConf.load(cfg_path)
        return cfg
    except FileNotFoundError:
        raise ValueError(
            f"Task '{task}' not supported (no config found at {str(cfg_path)})."
        )


@hydra.main(version_base=None, config_path="conf", config_name="main")
@log_exceptions(logger)
def main(cfg: DictConfig) -> None:
    task: str = cfg.task
    task_fun: Callable[[DictConfig], None] = TASK_FUNS.get(task)
    if task_fun is None:
        raise ValueError(f"Task '{task}' not supported (no task function found).")

    task_cfg = load_default_task_cfg(task)
    # update the default task_cfg with the values from the user-specified cfg
    task_cfg.merge_with(cfg)
    task_fun(task_cfg)


if __name__ == "__main__":
    main()
