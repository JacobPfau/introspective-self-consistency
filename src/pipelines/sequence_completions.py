"""
This file contains functions for generating ambiguous sequence completition problems.

An ambiguous sequence completition problem.
Currently only integer sequences are implemented.

Use like:
>>> from src.pipelines.sequence_completions import find_ambiguous_integer_sequences
>>> ambiguous_integer_sequences = find_ambiguous_integer_sequences()

ToDo:
(A) Brainstorming sequence
xth index in a sequence satisfying ORs of one or more criterion
- add more criteria here (its just module first term or second term)
Binary sequences (bitshift, codes, ect) (though error detecting / correcting codes might be its own pipeline)
String sequences (concat, reverasal, rotations, lexicographic, substitions, string progresions, ect.)
(B) Prompt variation
Generating few-shot templtes automatticaly with few-shot types:
- Oracle: Sample the true underlying rule at different non-overlapping steps or different terms.
- Adversarial
    - Sample ambiguous sequences
    - Sample sequences that **do not include similar rules**
- Samples with ambiguity that show potential options.

(C) Evaluation & Experimental Design
Exact equality
- Function generates the next sequence value
- More capabilties check
- Filter out ambigious functions
- Ambigious functions manually have generate a certain number of steps and check they eventually don't generate functions
Consistency evaluator - two outputs consistent
- Non-ambigious rules - success rate - in generateing examples
- Ambigious rules - success rate - in generating examples


"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.models import BaseModel
from src.pipelines.classes import ShotSamplingType, TaskType
from src.pipelines.sequences import get_sequences_as_dict
from src.prompt_generation import PromptBase, get_formatted_prompt


def find_ambiguous_integer_sequences(
    max_constant_term_one: int = 4,
    max_constant_term_two: int = 4,
    num_steps_to_check: int = 4,
    step_offsets: int = 4,
    disambiguate: bool = False,
    disambiguate_steps: int = 4,
    track_generating_fns: bool = False,
    multiple_offsets: bool = True,
) -> Dict[str, List[Dict[str, Union[str, int]]]]:
    """
    Find ambiguous sequence using brute force search
    over a set of progressions.

    A sequence is said to ambiguous if an initial set of two or more completions
    can be generated by different functions.

    Args:
        disambiguate: If True, only includes ambiguous function pairs for which there exists a disambiguating index.
        disambiguate_steps: The number of indices to check for disambiguation after finding an ambiguous pair.
        track_generating_fns: If True, includes the generating functions in the output dict.

    Returns a dict of ambiguous sequence objects:
    - key: ambiguous_sequence
    - value: list of function_candidates with their offsets.

    E.g.
    '1,3': [
        {'fn': 'lambda x: 3 ** (1*x)', 'offset': 0},
        {'fn': 'lambda x: (2*x) + 1', 'offset': 0}
    ]
    """
    progression_base_fns = get_sequences_as_dict()
    progressions_to_check = set()
    for const_term_one in range(max_constant_term_one):
        for const_term_two in range(max_constant_term_two):
            for name, progression in progression_base_fns.items():
                progressions_to_check.add(
                    (
                        (name, const_term_one, const_term_two),
                        progression.format(const_term_one + 1, const_term_two),
                    )
                )
    ambiguous_sequences = {}
    for ind, pair in enumerate(progressions_to_check):
        metadata_a, fn_a = pair
        for metadata_b, fn_b in list(progressions_to_check)[ind + 1 :]:
            if fn_a == fn_b:
                continue

            # check the sequence progressions
            # through n steps and add to ambiguous_sequences
            # if ambiguous
            _check_ambiguity(
                num_steps_to_check,
                step_offsets,
                ambiguous_sequences,
                fn_a,
                fn_b,
                metadata_a,
                metadata_b,
                disambiguate,
                disambiguate_steps,
                track_generating_fns,
                multiple_offsets,
            )
    return ambiguous_sequences


def _check_ambiguity(
    num_steps_to_check: int,
    step_offsets: int,
    ambiguous_sequences: dict,
    fn_a: str,
    fn_b: str,
    metadata_a: Tuple[str, int, int],
    metadata_b: Tuple[str, int, int],
    disambiguate: bool,
    disambiguate_steps: int,
    track_generating_fns: bool,
    multiple_offsets: bool,
) -> None:
    """
    Check the ambiguity of two sequence generating function

    We step through num_steps_to_check.
    As we do we check if fn_a(step) == fn_b(step) and
    fn_a(step + 1) != fn_b(step + 1)

    We account for unaligned sequences by iterating through offsets.

    !This function mutates ambiguous_sequences and adds ambiguous sequences
    as it goes.
    """

    for step_a_offset in range(step_offsets):
        for step_b_offset in range(step_offsets):
            completions = []
            if disambiguate:
                seq_acceptable = 0  # tracks if the sequence is ambiguous and (if disambiguate=True) if it is disambiguate-able.
                # 0 if not ambiguous/not disambiguate-able. Otherwise stores index of disambiguating value
            else:
                seq_acceptable = 1
            for step in range(num_steps_to_check):
                fn_a_step = eval(fn_a)(step + step_a_offset)
                fn_b_step = eval(fn_b)(step + step_b_offset)
                if fn_a_step != fn_b_step:
                    seq_acceptable = 0
                    break

                completions.append(fn_a_step)

                # if the next step are the same
                # they are not ambiguous: continue.
                fn_a_step = eval(fn_a)(step + 1 + step_a_offset)
                fn_b_step = eval(fn_b)(step + 1 + step_b_offset)

                if fn_a_step == fn_b_step:
                    continue
                elif fn_a_step != fn_b_step and step == num_steps_to_check - 1:
                    seq_acceptable = 1
            # if we have a sequence that isn't all the same
            # and is more than one we found an ambiguous sequence
            if (
                seq_acceptable
                and len(set(completions)) > 1
                and len([comp for comp in completions if comp != 0]) > 1
            ):
                if disambiguate:
                    # check if there is a disambiguating index
                    # if there is not, continue
                    for step in range(
                        num_steps_to_check, num_steps_to_check + disambiguate_steps
                    ):
                        fn_a_step = eval(fn_a)(step + step_a_offset)
                        fn_b_step = eval(fn_b)(step + step_b_offset)
                        if fn_a_step != fn_b_step:
                            seq_acceptable = step
                            break
                        else:
                            seq_acceptable = 0
                if seq_acceptable:
                    seq_string = ",".join([str(comp) for comp in completions])
                    if seq_string not in ambiguous_sequences:
                        ambiguous_sequences[seq_string] = []

                    fn_a_item = {
                        "fn": fn_a,
                        "offset": step_a_offset,
                        "metadata": metadata_a,
                    }
                    fn_b_item = {
                        "fn": fn_b,
                        "offset": step_b_offset,
                        "metadata": metadata_b,
                    }
                    if fn_a_item not in ambiguous_sequences[seq_string]:
                        if not track_generating_fns:
                            ambiguous_sequences[seq_string].append(fn_a_item)
                        else:
                            ambiguous_sequences[seq_string].append(
                                {
                                    "data": fn_a_item,
                                    "metadata": metadata_a,
                                    "disambiguating_pair_data": fn_b_item,
                                    "disambiguating_pair_metadata": metadata_b,
                                    "disambiguating_step": seq_acceptable,
                                }
                            )

                    if fn_b_item not in ambiguous_sequences[seq_string]:
                        if not track_generating_fns:
                            ambiguous_sequences[seq_string].append(fn_b_item)
                    if not multiple_offsets:
                        return


def generate_shot_pool(
    n_shots: int = 8,
    base_fn: dict = None,
    shot_type: ShotSamplingType = ShotSamplingType.RANDOM,
    ambiguous_sequences: dict = None,
) -> List[Dict[str, Any]]:
    """Generate a pool of `n_shots` of candidate functions.
    Depending on `shot_type` and the `base_fn`, candidates are sampled
    from a certain type of function.
    Note: the returned candidates may not necessarily lead to ambiguous sequences.
    """

    fn_pool = []
    if shot_type == ShotSamplingType.RANDOM:
        fn_pool = list(get_sequences_as_dict().values())
    elif shot_type == ShotSamplingType.SAME_CLASS:
        fn_pool = list(
            seq_fn
            for seq_key, seq_fn in get_sequences_as_dict().items()
            if seq_key == base_fn["metadata"][0]
        )
    elif shot_type == ShotSamplingType.EXCLUDE_CLASS:
        fn_pool = list(
            seq_fn
            for seq_key, seq_fn in get_sequences_as_dict().items()
            if seq_key != base_fn["metadata"][0]
        )

    shot_pool = []
    # we generate a prompt_pool with random parameters
    pool_size = 2 * n_shots
    if shot_type in [
        ShotSamplingType.RANDOM,
        ShotSamplingType.SAME_CLASS,
        ShotSamplingType.EXCLUDE_CLASS,
    ]:
        for _ in range(pool_size):
            offset = random.randint(0, 3)
            first_term = random.randint(1, 5)
            second_term = random.randint(0, 4)
            fn = random.choice(list(fn_pool))
            shot_pool.append(
                {"fn": fn.format(first_term, second_term), "offset": offset}
            )

    elif shot_type == ShotSamplingType.SAME_FN:
        for _ in range(pool_size):
            offset = random.randint(0, 10)
            shot_pool.append({"fn": base_fn["fn"], "offset": offset})

    elif shot_type == ShotSamplingType.AMBIGUOUS:
        while len(shot_pool) < pool_size:
            fn_item = random.choice(list(ambiguous_sequences.items()))
            fns = np.random.choice(fn_item[1], 2, replace=False)
            shot_pool.extend(fns)

    shots = np.random.choice(shot_pool, size=n_shots, replace=False)
    # continue to draw if fn_item in shots
    while base_fn in shots:
        shots = np.random.choice(shot_pool, size=n_shots, replace=False)
    return shots


def resolve_fn(fn_item: dict, step: int) -> int:
    # resolve function to a given step
    step = step + fn_item["offset"]
    return eval(fn_item["fn"])(step)


def _create_sequence_prompt(
    sequence: str,
    fn_item: dict,
    task_type: TaskType,
    use_multiple_choice=False,
) -> Tuple[str, str]:
    """Creates a prompt for completion type or explanation type prompts

    Example:
    >>> _create_sequence_prompt("1,25", {"fn": 'lambda x: 5 ** (2 * x)', 'offset': 0}, "explanation")
    ('\nFor the sequence: 1,25\n\nGive the code that generates the above sequence.\n', 'lambda x: 5 ** (2 * x)', '')
    >>> _create_sequence_prompt("1,25", {"fn": 'lambda x: 5 ** (2 * x)', 'offset': 0}, "completion")
    ('\nFor the sequence: 1,25\n\nComplete the next number and only the next number.\n', 625, '')

    Args:
        sequence (str): the sequence
        fn_item (dict): the fn_item
        task_type (TaskType): the task type (completion or explanation)

    Returns:
        Tuple[str, str]: prompt, completion
    """

    if isinstance(task_type, str):
        task_type = TaskType(task_type)

    if task_type == TaskType.COMPLETION:
        prompt = get_formatted_prompt(PromptBase.BASE_COMPLETION, {"seq": sequence})
        last_step = len(sequence.split(","))
        completion = str(resolve_fn(fn_item, last_step))
    elif task_type == TaskType.EXPLANATION:

        if use_multiple_choice:
            prompt = get_formatted_prompt(
                PromptBase.EXPLANATION_MULTIPLE_CHOICE, {"seq": sequence}
            )
        else:
            prompt = get_formatted_prompt(
                PromptBase.BASE_EXPLANATION, {"seq": sequence}
            )
        completion = fn_item["fn"]
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    return prompt, completion


def _sample_shots(
    sequence: str,
    fn_item: dict,
    n_shots: int,
    task_type: TaskType = TaskType.COMPLETION,
    n_mc_options: Optional[int] = None,
    shot_type: ShotSamplingType = ShotSamplingType.RANDOM,
    ambiguous_sequences: dict = None,
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
        prompt, answer = _create_sequence_prompt(
            sequence,
            shot,
            task_type,
            use_multiple_choice=False if n_mc_options is None else True,
        )

        if n_mc_options is not None:
            # sample random incorrect explanations
            multi_choice_options = [fn_item]
            while len(multi_choice_options) < n_mc_options:
                shots = generate_shot_pool(
                    n_shots=n_mc_options,
                    base_fn=fn_item,
                    shot_type=shot_type,
                    ambiguous_sequences=ambiguous_sequences,
                )
                for shot in shots:
                    if shot["fn"] != fn_item["fn"] and shot not in multi_choice_options:
                        multi_choice_options.append(shot)
                    if len(multi_choice_options) == n_mc_options:
                        break

            random.shuffle(multi_choice_options)  # shuffle options to avoid bias
            answer = (
                multi_choice_options.index(fn_item) + 1
            )  # when not using multiple choice, store the correct answer
            for i, expl in enumerate(multi_choice_options):
                prompt += f"{i+1}. {expl['fn']}\n"

        prompts.append({"prompt": prompt, "answer": answer})

    return prompts


def get_sequence_string(shot, steps) -> str:
    """Given a function and number of steps, generate a sequence string separated by commas"""
    sequence = ",".join([str(resolve_fn(shot, step)) for step in range(steps)])
    return sequence


def generate_sequence_completion_prompt(
    sequence: str,
    fn_item: dict,
    task_type=TaskType.COMPLETION,
    n_shots: int = 0,
    shot_type=ShotSamplingType.RANDOM,
    ambiguous_sequences: dict = None,
) -> dict:
    """
    Generate sequence completion prompts
    including support for few_shot with `:n_shots`

    Returns:
        dict:
    """

    if isinstance(task_type, str):
        task_type = TaskType(task_type)
    if isinstance(shot_type, str):
        shot_type = ShotSamplingType(shot_type)

    prompt_turns = [
        {
            "role": "system",
            "content": get_formatted_prompt(PromptBase.SYSTEM_FUNCTION_SPACE),
        },
    ]

    shots = _sample_shots(
        sequence,
        fn_item,
        task_type=task_type,
        n_shots=n_shots,
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

    prompt, answer = _create_sequence_prompt(sequence, fn_item, task_type)

    prompt_turns.append({"role": "user", "content": prompt})
    return {"prompt_turns": prompt_turns, "answer": answer}


def generate_sequence_explanation_prompt_with_multiple_choices(
    sequence: str,
    fn_item: dict,
    model: BaseModel,
    n_mc_options: int = 4,
    n_shots: int = 0,
    shot_type: Union[ShotSamplingType, str] = "random",
    ambiguous_sequences: dict = None,
) -> dict:

    task_type = TaskType.EXPLANATION
    prompt_turns = [
        {
            "role": "system",
            "content": get_formatted_prompt(PromptBase.SYSTEM_FUNCTION_SPACE),
        },
    ]

    if isinstance(shot_type, str):
        shot_type = ShotSamplingType(shot_type)

    shots = _sample_shots(
        sequence,
        fn_item,
        task_type=TaskType.EXPLANATION,
        n_shots=n_shots,
        n_mc_options=n_mc_options,
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

    prompt, answer = _create_sequence_prompt(
        sequence, fn_item, task_type, use_multiple_choice=True
    )

    prompt_turns.append({"role": "user", "content": prompt})
    return {"prompt_turns": prompt_turns, "answer": answer}
