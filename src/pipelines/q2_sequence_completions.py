import random
from typing import Any, Dict, List, Optional, Tuple

from src.models import BaseModel
from src.pipelines import TaskType
from src.prompt_generation import PromptBase, get_formatted_prompt

from .sequence_completions import resolve_fn


def _create_sequence_prompt(
    sequence: str,
    valid_fns: List[dict],
    task_type: TaskType,
    use_multiple_choice=False,
    max_considerations: Optional[int] = None,
    model: Optional[BaseModel] = None,
) -> Tuple[str, str, str]:
    """Creates a prompt for completion type or explanation type prompts"""

    prompt = ""
    answer = ""
    if task_type == TaskType.COMPLETION:
        prompt = get_formatted_prompt(PromptBase.POSSIBLE_COMPLETION, {"seq": sequence})
        if max_considerations is not None and model is not None:

            prompt += get_formatted_prompt(
                PromptBase.CONSIDERATIONS,
                {"n_consider": max_considerations, "model_name": model.value},
            )

            # roll out valid fns to obtain valid completions
            # TODO: replace with sequence roll out function
            last_step = len(sequence.split(","))
            valid_answers = [resolve_fn(fn_item, last_step) for fn_item in valid_fns]

            # separate valid answers properly
            valid_answers = list(set(valid_answers))
            random.shuffle(valid_answers)
            answer = "\\n".join(
                [str(ans) for ans in valid_answers[:max_considerations]]
            )
        elif use_multiple_choice:
            raise NotImplementedError("No multiple choice for completion prompts")
        else:
            raise NotImplementedError()

    elif task_type == TaskType.EXPLANATION:

        if use_multiple_choice:
            raise NotImplementedError()
        elif max_considerations is not None and model is not None:
            prompt = get_formatted_prompt(
                PromptBase.BASE_EXPLANATION, {"seq": sequence}
            )[
                :-2
            ]  # leave out the last period
            prompt += get_formatted_prompt(
                PromptBase.CONSIDERATIONS,
                {"n_consider": max_considerations, "model_name": model.value},
            )

            # separate valid answers properly
            valid_fns = list({fn_item["fn"] for fn_item in valid_fns})
            random.shuffle(valid_fns)
            answer = "\\n".join(valid_fns[:max_considerations])
        else:
            raise NotImplementedError()

    return prompt, answer


def _sample_shot_pool_from_amb_seqs(
    ambiguous_sequences: dict,
    sequence: str,
    n_shots: int = 8,
) -> Dict[str, Any]:
    """Generate a pool of `n_shots` of candidate ambiguous sequences and functions.
    Candidates are sampled randomly from the set of ambiguous sequences.
    Explicitly excludes the base `sequence` from the pool.
    """

    if n_shots > len(ambiguous_sequences):
        raise ValueError(
            f"Number of shots {n_shots} is larger than the number of ambiguous sequences {len(ambiguous_sequences)}"
        )

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
    task_type: TaskType = TaskType.COMPLETION,
    max_considerations: int = 5,
) -> List[Dict[str, Any]]:
    """
    Sample `:n_shots` number of shots and construct a prompt.
    Initially we randomly generate `:_generate_shot_pool` the shots.
    """

    # sample shot pool from ambiguous sequences
    shots = _sample_shot_pool_from_amb_seqs(
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
            task_type,
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
    task_type: TaskType = TaskType.COMPLETION,
    n_shots: int = 0,
    max_considerations: int = 5,
    model: Optional[BaseModel] = None,
) -> dict:
    """
    Generate sequence completion prompts with in-context examples listing possible, valid continuations
    including support for few_shot with `:n_shots`

    Args:
        sequence (str): The input sequence to be completed.
        valid_fns (List[dict]): A list of valid functions that can be used to complete the sequence.
        ambiguous_sequences (dict): A dictionary of ambiguous sequences and their possible completions.
        task_type (TaskType, optional): The type of task to perform. Defaults to TaskType.COMPLETION.
        n_shots (int, optional): The number of few-shot examples to generate. Defaults to 0.
        max_considerations (int, optional): The maximum number of possible completions to consider. Defaults to 5.
        model (Optional[BaseModel], optional): The model to use for generating completions. Defaults to None.

    Returns:
        dict:
    """

    prompt_turns = [
        {
            "role": "system",
            "content": get_formatted_prompt(PromptBase.SYSTEM_FUNCTION_SPACE),
        },
    ]

    shots = _sample_shots_with_considerations(
        sequence,
        n_shots=n_shots,
        max_considerations=max_considerations,
        model=model,
        task_type=task_type,
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
        task_type=task_type,
        use_multiple_choice=False,
        max_considerations=max_considerations,
        model=model,
    )

    prompt_turns.append({"role": "user", "content": prompt})
    return {"prompt_turns": prompt_turns, "answer": answer}
