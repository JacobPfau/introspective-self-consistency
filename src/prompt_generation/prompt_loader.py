from enum import Enum


class PromptBase(Enum):
    BASE_COMPLETION = "base_completion"
    BASE_EXPLANATION = "base_explanation"
    EXPLANATION_MULTIPLE_CHOICE = "base_explanation_multiple_choice"
    CONSIDERATIONS = "consideration"
    SYSTEM_MATH = "system_math"
    BASE_CONSISTENCY = "base_consistency"
    CONSISTENCY_COMPLETION = "consistency_completion"


def _load_base_prompt_from_txt(txt_file: str) -> str:
    with open(txt_file) as f:
        return f.read()


def get_formatted_prompt(prompt_base: PromptBase, kw_args: dict = {}) -> str:

    if prompt_base == PromptBase.BASE_COMPLETION:
        path = "src/prompt_generation/prompts_txt/base_completion.txt"
    elif prompt_base == PromptBase.BASE_EXPLANATION:
        path = "src/prompt_generation/prompts_txt/base_explanation.txt"
    elif prompt_base == PromptBase.EXPLANATION_MULTIPLE_CHOICE:
        path = "src/prompt_generation/prompts_txt/explanation_mc.txt"
    elif prompt_base == PromptBase.CONSIDERATIONS:
        path = "src/prompt_generation/prompts_txt/consider.txt"
    elif prompt_base == PromptBase.SYSTEM_MATH:
        path = "src/prompt_generation/prompts_txt/system_math.txt"
    elif prompt_base == PromptBase.BASE_CONSISTENCY:
        path = "src/prompt_generation/prompts_txt/base_consistency.txt"
    elif prompt_base == PromptBase.CONSISTENCY_COMPLETION:
        path = "src/prompt_generation/prompts_txt/consistency_completion.txt"
    else:
        raise ValueError(f"Invalid prompt base: {prompt_base}")

    base_prompt = _load_base_prompt_from_txt(path)

    try:
        return base_prompt.format(**kw_args)
    except Exception as e:
        raise ValueError(f"Invalid args for prompt base: {prompt_base}") from e
