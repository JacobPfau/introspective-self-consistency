import logging
import random
from pathlib import Path
from typing import Callable, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig

from src.evals.config import (
    BaseEvalConfig,
    Q21LogprobInequalityConfig,
    Q22ModelVerbalizationConfig,
    SequenceCompletionBaseChangeConfig,
    SequenceCompletionCapabilityConfig,
    SequenceCompletionEqConfig,
)
from src.evals.q2_1_logprob_inequality import run_q2_1_eval
from src.evals.q2_2_alternative_verbalization import run_q2_2_eval
from src.evals.sequence_completion import evaluate_sequence_completion_equality
from src.evals.sequence_completion_capability import (
    evaluate_sequence_completion_capability,
)
from src.evals.sequence_completion_with_base_change import (
    evaluate_compute_dependence_with_base_changes,
)
from src.utils import log_exceptions

logger = logging.getLogger(__name__)

TASK_FUNS = {
    "sequence_completion_capability": {
        "fn": evaluate_sequence_completion_capability,
        "config": SequenceCompletionCapabilityConfig,
    },
    "sequence_completion_equality": {
        "fn": evaluate_sequence_completion_equality,
        "config": SequenceCompletionEqConfig,
    },
    "compute_dependence_with_base_changes": {
        "fn": evaluate_compute_dependence_with_base_changes,
        "config": SequenceCompletionBaseChangeConfig,
    },
    "q2_1_logprob_inequality": {
        "fn": run_q2_1_eval,
        "config": Q21LogprobInequalityConfig,
    },
    "q2_2_alternative_verbalization": {
        "fn": run_q2_2_eval,
        "config": Q22ModelVerbalizationConfig,
    },
}


_DEFAULT_RANDOM_SEED = 21


def _seed_everything(seed=_DEFAULT_RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


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
    _seed_everything(task_cfg.seed)
    task_fun(task_cfg)
    logger.info(f"Output dir: {str(Path.cwd())}")


if __name__ == "__main__":
    main()
