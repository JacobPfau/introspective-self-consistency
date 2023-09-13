import random
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.models import BaseModel
from src.prompt_generation.alternative_considerations import (
    BASE_CONSIDER_PROMPT,
    BASE_PROMPT_EXPLANATION_MULTIPLE_CHOICE,
    MODEL_PRIMING_PROMPT,
)

from ..prompt_generation.base_prompts import (
    BASE_PROMPT,
    BASE_PROMPT_COMPLETION,
    BASE_PROMPT_EXPLANATION,
    SYSTEM_PROMPT,
)
from .sequence_completions import resolve_fn


def _create_sequence_prompt(
    sequence: str,
    valid_fns: List[dict],
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
            prompt += BASE_PROMPT_COMPLETION[:-2]  # leave out the last period
            prompt += MODEL_PRIMING_PROMPT.format(model.value)
            prompt += BASE_CONSIDER_PROMPT.format(max_considerations)

            # roll out valid fns to obtain valid completions
            last_step = len(sequence.split(","))
            valid_answers = [resolve_fn(fn_item, last_step) for fn_item in valid_fns]

            # separate valid answers properly
            valid_answers = list(set(valid_answers))
            random.shuffle(valid_answers)
            answer = "\\n".join(
                [str(ans) for ans in valid_answers[:max_considerations]]
            )

        else:
            raise NotImplementedError()

    elif prompt_type == "explanation":

        if use_multiple_choice:
            prompt += BASE_PROMPT_EXPLANATION_MULTIPLE_CHOICE
            raise NotImplementedError()
        elif max_considerations is not None and model is not None:
            prompt += BASE_PROMPT_EXPLANATION[:-2]  # leave out the last period
            prompt += MODEL_PRIMING_PROMPT.format(model.value)
            prompt += BASE_CONSIDER_PROMPT.format(max_considerations)

            # separate valid answers properly
            valid_fns = list({fn_item["fn"] for fn_item in valid_fns})
            random.shuffle(valid_fns)
            answer = "\\n".join(valid_fns[:max_considerations])
        else:
            raise NotImplementedError()

    return prompt, answer


def sample_shot_pool_from_amb_seqs(
    ambiguous_sequences: dict,
    sequence: str,
    n_shots: int = 8,
) -> Dict[str, Any]:
    """Generate a pool of `n_shots` of candidate ambiguous sequences and functions.
    Candidates are sampled randomly from the set of ambiguous sequences
    """

    shot_pool = {
        k: ambiguous_sequences[k]
        for k in random.sample(ambiguous_sequences.keys(), n_shots)
    }
    # continue to draw if base sequence in shot pool
    while sequence in shot_pool:
        shot_pool = {
            k: ambiguous_sequences[k]
            for k in random.sample(ambiguous_sequences.keys(), n_shots)
        }

    return shot_pool


def _sample_shots_with_considerations(
    sequence: str,
    n_shots: int,
    model: BaseModel,
    ambiguous_sequences: dict,
    prompt_type: Literal["completion", "explanation"] = "completion",
    max_considerations: int = 5,
) -> List[Dict[str, Any]]:
    """
    Sample `:n_shots` number of shots and construct a prompt.
    Initially we randomly generate `:_generate_shot_pool` the shots.
    """

    # sample shot pool from ambiguous sequences
    shots = sample_shot_pool_from_amb_seqs(
        ambiguous_sequences=ambiguous_sequences,
        sequence=sequence,
        n_shots=n_shots,
    )
    # for all the shots create sequence prompts
    prompts = []
    for seq_key, seq_fns in shots.items():
        prompt, answer = _create_sequence_prompt(
            seq_key,
            seq_fns,
            prompt_type,
            use_multiple_choice=False,
            max_considerations=max_considerations,
            model=model,
        )

        prompts.append({"prompt": prompt, "answer": answer})

    return prompts


def generate_sequence_completion_prompt_with_valid_continuations(
    sequence: str,
    valid_fns: List[dict],
    ambiguous_sequences: dict,
    prompt_type: Literal["completion", "explanation"] = "completion",
    n_shots: int = 0,
    max_considerations: int = 5,
    model: Optional[BaseModel] = None,
) -> dict:
    """
    Generate sequence completion prompts with in-context examples listing possible, valid continuations
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
        sequence,
        n_shots=n_shots,
        max_considerations=max_considerations,
        model=model,
        prompt_type=prompt_type,
        ambiguous_sequences=ambiguous_sequences,
    )

    for shot in shots:
        answer = str(shot["answer"])

        turns = [
            {"role": "user", "content": shot["prompt"]},
            {"role": "assistant", "content": answer},
        ]
        prompt_turns.extend(turns)

    prompt, answer = _create_sequence_prompt(
        sequence,
        valid_fns=valid_fns,
        prompt_type=prompt_type,
        use_multiple_choice=False,
        max_considerations=max_considerations,
        model=model,
    )

    prompt_turns.append({"role": "user", "content": prompt})
    return {"prompt_turns": prompt_turns, "answer": answer}
