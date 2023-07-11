import random
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.models.base_model import BaseModel

from .alternative_completions import get_valid_alternative_funcs
from .sequence_completions import (
    BASE_PROMPT,
    BASE_PROMPT_COMPLETION,
    BASE_PROMPT_EXPLANATION,
    SYSTEM_PROMPT,
    generate_shot_pool,
    get_sequence_string,
    resolve_fn,
)

PromptType = Literal["random", "same_fn", "same_class", "ambigious", "exclude_class"]


BASE_CONSIDER_PROMPT = """
Consider up to {} possible and valid answers separated by escape character '\\n' "
"""

MODEL_PRIMING_PROMPT = ", as determined by you, {}."

BASE_PROMPT_EXPLANATION_MULTIPLE_CHOICE = """
Select the code that generates the above sequence from the following options.
Only respond with the number of the valid option.
Options:
"""


def _create_sequence_prompt(
    sequence: str,
    fn_item: dict,
    valid_answers: List[dict],
    prompt_type: Literal["completion", "explanation"],
    use_multiple_choice=False,
    max_considerations: Optional[int] = None,
    model: Optional[BaseModel] = None,
) -> Tuple[str, str, str]:
    """Creates a prompt for completion type or explanation type prompts

    Example:
    >>> _create_sequence_prompt("1,25", {"fn": 'lambda x: 5 ** (2 * x)', 'offset': 0}, "explanation", use_cot=False)
    ('\nFor the sequence: 1,25\n\nGive the code that generates the above sequence.\n', 'lambda x: 5 ** (2 * x)', '')
    >>> _create_sequence_prompt("1,25", {"fn": 'lambda x: 5 ** (2 * x)', 'offset': 0}, "completion", use_cot=False)
    ('\nFor the sequence: 1,25\n\nComplete the next number and only the next number.\n', 625, '')

    Args:
        sequence (str): the sequence
        fn_item (dict): the fn_item
        prompt_type (Literal[&quot;completion&quot;, &quot;explanation&quot;]): type of prompt to use
        use_cot (bool, optional)

    Returns:
        Tuple[str, str, str]: prompt, compleition, and chain of thought
    """
    prompt = BASE_PROMPT.format(sequence)
    answer = ""
    if prompt_type == "completion":

        if use_multiple_choice:
            prompt += BASE_PROMPT_COMPLETION
        elif max_considerations is not None and model is not None:
            prompt += BASE_PROMPT_COMPLETION[:-1]  # leave out the last period
            prompt += MODEL_PRIMING_PROMPT.format(model.value)
            prompt += BASE_CONSIDER_PROMPT.format(max_considerations)

            # roll out valid fns to obtain valid completions
            last_step = len(sequence.split(","))
            valid_answers = [
                resolve_fn(fn_item, last_step) for fn_item in valid_answers
            ]

            # separate valid answers properly
            random.shuffle(valid_answers)
            answer = "\\n".join(valid_answers[:max_considerations])

        else:
            raise NotImplementedError()

    elif prompt_type == "explanation":

        if use_multiple_choice:
            prompt += BASE_PROMPT_EXPLANATION_MULTIPLE_CHOICE
            answer = fn_item["fn"]
        elif max_considerations is not None and model is not None:
            prompt += BASE_PROMPT_EXPLANATION[:-1]  # leave out the last period
            prompt += MODEL_PRIMING_PROMPT.format(model.value)
            prompt += BASE_CONSIDER_PROMPT.format(max_considerations)

            # separate valid answers properly
            random.shuffle(valid_answers)
            answer = "\\n".join(valid_answers[:max_considerations])
        else:
            raise NotImplementedError()

    return prompt, answer


def _sample_shots_with_considerations(
    fn_item: dict,
    n_shots: int,
    model: BaseModel,
    ambiguous_sequences: dict,
    prompt_type: Literal["completion", "explanation"] = "completion",
    max_considerations: int = 5,
    shot_type: Literal[
        "random", "same_fn", "same_class", "ambigious", "exclude_class"
    ] = "few_shot",
) -> List[Dict[str, Any]]:
    """
    Sample `:n_shots` number of shots and construct a prompt.
    Initially we randomly generate `:_generate_shot_pool` the shots.
    """
    shots = generate_shot_pool(
        n_shots=n_shots,
        base_fn=fn_item,
        shot_type=shot_type,
        ambiguous_sequences=ambiguous_sequences,
    )
    # for all the shots create sequence prompts
    prompts = []
    for shot in shots:
        steps = random.randint(2, 4)
        sequence = get_sequence_string(shot, steps)

        # get valid alternative for this example function and sequence
        _, valid_alternatives = get_valid_alternative_funcs(
            shot,
            ambiguous_sequences,
            num_valid=max_considerations,
            org_seq=sequence,
        )

        prompt, answer = _create_sequence_prompt(
            sequence,
            shot,
            valid_alternatives,
            prompt_type,
            use_multiple_choice=False,
            max_considerations=max_considerations,
            model=model,
        )

        prompts.append({"prompt": prompt, "answer": answer})

    return prompts


def generate_sequence_completion_prompt_with_valid_continuations(
    sequence: str,
    fn_item: dict,
    valid_fns: List[dict],
    ambiguous_sequences: dict,
    prompt_type: Literal["completion", "explanation"] = "completion",
    n_shots: int = 0,
    max_considerations: int = 5,
    shot_type: PromptType = "random",
    model: Optional[BaseModel] = None,
) -> dict:
    """
    Generate sequence completion prompts where in-context examples list possible, valid continuations
    including support for few_shot with `:n_shots`
    and chain of thought step completions with `:use_cot`

    Returns:
        dict:
    """
    # TODO: this should be generic so it isn't coupled to ChatGPT
    prompt_turns = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
    ]

    shots = _sample_shots_with_considerations(
        fn_item,
        n_shots=n_shots,
        max_considerations=max_considerations,
        model=model,
        prompt_type=prompt_type,
        shot_type=shot_type,
        ambiguous_sequences=ambiguous_sequences,
    )

    for shot in shots:
        answer = str(shot["answer"])

        turns = [
            {"role": "user", "content": shot["prompt"]},
            {"role": "assistant", "content": answer},
        ]
        prompt_turns.extend(turns)

    prompt, answer, _ = _create_sequence_prompt(
        sequence,
        fn_item,
        valid_answers=valid_fns,
        prompt_type=prompt_type,
        max_considerations=max_considerations,
        model=model,
    )

    prompt_turns.append({"role": "user", "content": prompt})
    return {"prompt_turns": prompt_turns, "answer": answer}
