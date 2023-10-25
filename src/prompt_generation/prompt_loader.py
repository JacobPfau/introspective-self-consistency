import os
from enum import Enum

from hydra.utils import get_original_cwd

from src.pipelines.sequences import get_sequences_as_dict


class PromptBase(Enum):
    BASE_COMPLETION = "base_completion"
    BASE_EXPLANATION = "base_explanation"
    EXPLANATION_MULTIPLE_CHOICE = "base_explanation_multiple_choice"
    CONSIDERATIONS = "consideration"
    SYSTEM_MATH = "system_math"
    SYSTEM_FUNCTION_SPACE = "system_function_space"
    BASE_CONSISTENCY = "base_consistency"
    CONSISTENCY_COMPLETION = "consistency_completion"
    POSSIBLE_COMPLETION = "possible_completion"
    ROBUST_COMPLETION_BASE10 = "robust_completion_base10"
    ROBUST_COMPLETION_BASE2 = "robust_completion_base2"
    ROBUST_EXPLANATION_BASE10 = "robust_explanation_base10"
    ROBUST_EXPLANATION_BASE2 = "robust_explanation_base2"
    ROLE_PROMPT = "role_prompt"
    COMPLETION_SKELETON_TEXT = "completion_skeleton_text"
    EXPALANATION_SKELETON_TEXT = "explanation_skeleton_text"
    COMPLETION_SHOT_TEXT = "completion_shot_text"
    EXPLANATION_SHOT_TEXT = "explanation_shot_text"


def _load_base_prompt_from_txt(txt_file: str) -> str:
    with open(txt_file) as f:
        return f.read()


def get_formatted_prompt(prompt_base: PromptBase, kw_args: dict = {}) -> str:

    root_dir = get_original_cwd()  # get the root directory when using hydra

    if prompt_base == PromptBase.BASE_COMPLETION:
        path = os.path.join(
            root_dir, "src/prompt_generation/prompts_txt/base_completion.txt"
        )
    elif prompt_base == PromptBase.BASE_EXPLANATION:
        path = os.path.join(
            root_dir, "src/prompt_generation/prompts_txt/base_explanation.txt"
        )
    elif prompt_base == PromptBase.EXPLANATION_MULTIPLE_CHOICE:
        path = os.path.join(
            root_dir, "src/prompt_generation/prompts_txt/explanation_mc.txt"
        )
    elif prompt_base == PromptBase.CONSIDERATIONS:
        path = os.path.join(root_dir, "src/prompt_generation/prompts_txt/consider.txt")
    elif prompt_base == PromptBase.SYSTEM_MATH:
        path = os.path.join(
            root_dir, "src/prompt_generation/prompts_txt/system_math.txt"
        )
    elif prompt_base == PromptBase.POSSIBLE_COMPLETION:
        path = os.path.join(
            root_dir, "src/prompt_generation/prompts_txt/possible_completion.txt"
        )
    elif prompt_base == PromptBase.BASE_CONSISTENCY:
        path = os.path.join(
            root_dir, "src/prompt_generation/prompts_txt/base_consistency.txt"
        )
    elif prompt_base == PromptBase.CONSISTENCY_COMPLETION:
        path = os.path.join(
            root_dir, "src/prompt_generation/prompts_txt/consistency_completion.txt"
        )
    elif prompt_base == PromptBase.SYSTEM_FUNCTION_SPACE:
        path = os.path.join(
            root_dir, "src/prompt_generation/prompts_txt/system_function_space.txt"
        )
        base_prompt = _load_base_prompt_from_txt(path)

        # progressions have two terms A and B
        for fn_name, base_fn in get_sequences_as_dict().items():
            base_prompt += "- " + fn_name + " -> " + base_fn.format("a", "b") + " \n"

        return base_prompt

    elif prompt_base == PromptBase.ROBUST_COMPLETION_BASE10:
        path = os.path.join(
            root_dir,
            "src/prompt_generation/prompts_txt/robustness_system_prompt_cont_10.txt",
        )
    elif prompt_base == PromptBase.ROBUST_COMPLETION_BASE2:
        path = os.path.join(
            root_dir,
            "src/prompt_generation/prompts_txt/robustness_system_prompt_cont_2.txt",
        )
    elif prompt_base == PromptBase.ROBUST_EXPLANATION_BASE10:
        path = os.path.join(
            root_dir,
            "src/prompt_generation/prompts_txt/robustness_system_prompt_exp_10.txt",
        )
    elif prompt_base == PromptBase.ROBUST_EXPLANATION_BASE2:
        path = os.path.join(
            root_dir,
            "src/prompt_generation/prompts_txt/robustness_system_prompt_exp_2.txt",
        )
    elif prompt_base == PromptBase.ROLE_PROMPT:
        path = os.path.join(
            root_dir, "src/prompt_generation/prompts_txt/role_prompt.txt"
        )
    elif prompt_base == PromptBase.COMPLETION_SKELETON_TEXT:
        path = os.path.join(
            root_dir, "src/prompt_generation/prompts_txt/continuation_skeleton_text.txt"
        )
    elif prompt_base == PromptBase.EXPALANATION_SKELETON_TEXT:
        path = os.path.join(
            root_dir, "src/prompt_generation/prompts_txt/explanation_skeleton.txt"
        )
    elif prompt_base == PromptBase.EXPLANATION_SHOT_TEXT:
        path = os.path.join(
            root_dir, "src/prompt_generation/prompts_txt/explanation_shot_text.txt"
        )
    elif prompt_base == PromptBase.COMPLETION_SHOT_TEXT:
        path = os.path.join(
            root_dir, "src/prompt_generation/prompts_txt/continuation_shot_text.txt"
        )
    else:
        raise ValueError(f"Invalid prompt base: {prompt_base}")

    base_prompt = _load_base_prompt_from_txt(path)

    try:
        return base_prompt.format(**kw_args)
    except Exception as e:
        raise ValueError(f"Invalid args for prompt base: {prompt_base}") from e
