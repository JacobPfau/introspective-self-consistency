import logging

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


@hydra.main(version_base=None, config_path="conf", config_name="main")
@log_exceptions(logger)
def main(cfg: DictConfig) -> None:
    if task_cfg := cfg.string_transformation_completion_equality:
        evaluate_string_transformation_equality(
            model=task_cfg.model,
            num_shots=task_cfg.num_shots,
            cot=task_cfg.use_cot,
        )

    if task_cfg := cfg.sequence_completion_equality:
        evaluate_sequence_completion_equality(
            model=task_cfg.model,
            max_offset=task_cfg.max_offset,
            num_shots=task_cfg.num_shots,
            cot=task_cfg.use_cot,
            few_shot_prompt_type=task_cfg.few_shot_prompt_type,
        )

    if task_cfg := cfg.compute_dependence_with_base_changes:
        evaluate_compute_dependence_with_base_changes(
            sequence_type=task_cfg.sequence_type,
            model=task_cfg.model,
            num_shots=task_cfg.num_shots,
            on_ambiguous_sequences=task_cfg.on_ambiguous_sequences,
            num_samples=task_cfg.num_samples,
        )

    if task_cfg := cfg.ambibench_category_prediction:
        evaluate_ambibench_category_prediction(
            model=task_cfg.model,
            data_dir=task_cfg.data_path,
            multiple_choice=task_cfg.multiple_choice,
        )

    if task_cfg := cfg.ambibench_completion:
        evaluate_ambibench_completion(
            model=task_cfg.model,
            data_dir=task_cfg.data_path,
        )


if __name__ == "__main__":
    main()
