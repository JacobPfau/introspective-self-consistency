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
Generate CoT:
- evaluate step by step with doctest (try before and after answer)
- resolve each term
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
from typing import Literal, Tuple

import numpy as np

SYSTEM_PROMPT = """
You are a mathematical assistant.
You are helping with integer sequences like arithmetic or geometric sequences.
The sequence's are not always 0 indexed, some are offset starting from an arbitrary i-index value.
You answer accurately and concisely.
"""

SYSTEM_PROMPT_COMPLETION = """
You only respond with numbers.
"""

SYSTEM_PROMPT_EXPLANATION = """
You only respond with code.
"""

BASE_PROMPT = """
For the sequence: {}
"""

BASE_PROMPT_COMPLETION = """
Complete the next number and only the next number.
"""

BASE_PROMPT_EXPLANATION = """
Give the code that generates the above sequence.
"""

COT_PROMPT = """
Let's solve this step by step:
"""

COT_STEP = """
Step {}:
>>> fn = {}; fn({})
{}
"""

# Integer sequence functions
sequence_functions = {
    "arithmetic_progression": "lambda x: ({} * x) + {}",
    "geometric_progression": "lambda x: ({} * x) * {}",
    "exponential_progression": "lambda x: ({} * x) ** {}",
    "power_progression": "lambda x: {} ** ({} * x)",
    "bit_or_progression": "lambda x: ({} * x) | {}",
    "modular_progression": "lambda x: (x * {}) % ({}+1)",
    "indexing_criteria_progression": (
        "lambda x: [i for i in range(100) if i % ({} + 1) or i % ({} + 1)][x]"
    ),
    "recursive_progression": (
        "(lambda a:lambda v:a(a,v))(lambda fn,x:1 if x==0 else {} * x * fn(fn,x-1) + {})"
    ),
}


def find_ambiguous_integer_sequences(
    max_constant_term_one: int = 4,
    max_constant_term_two: int = 4,
    num_steps_to_check: int = 4,
    step_offsets: int = 4,
    disambiguate: bool = False,
    disambiguate_steps: int = 4,
    track_generating_fns: bool = False,
    multiple_offsets: bool = True,
    valid_sequence_functions: dict = sequence_functions,
) -> dict:
    """
    Find ambiguous_integer_sequences using brute force search
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
    progression_base_fns = valid_sequence_functions
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
            check_ambiguity(
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


def check_ambiguity(
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
            seq_acceptable = 1  # tracks if the sequence is ambiguous and (if disambiguate=True) if it is disambiguate-able.
            # 0 if not ambiguous/not disambiguate-able. Otherwise stores index of disambiguating value
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


def _generate_shot_pool(
    n_shots: int = 8,
    base_fn: dict = None,
    shot_type: Literal[
        "random", "same_fn", "same_class", "ambigious", "exclude_class"
    ] = "random",
    ambiguous_sequences: dict = None,
):
    fn_pool = []
    if shot_type == "random":
        fn_pool = list(sequence_functions.values())
    elif shot_type == "same_class":
        fn_pool = list(
            seq_fn
            for seq_key, seq_fn in sequence_functions.items()
            if seq_key == base_fn["metadata"][0]
        )
    elif shot_type == "exclude_class":
        fn_pool = list(
            seq_fn
            for seq_key, seq_fn in sequence_functions.items()
            if seq_key != base_fn["metadata"][0]
        )

    shot_pool = []
    # we generate a prompt_pool with random parameters
    # TODO: move these magic strings to somewhere more visible
    pool_size = 2 * n_shots
    if shot_type in ["random", "same_class", "exclude_class"]:
        for _ in range(pool_size):
            offset = random.randint(0, 3)
            first_term = random.randint(1, 5)
            second_term = random.randint(0, 4)
            fn = random.choice(list(fn_pool))
            shot_pool.append(
                {"fn": fn.format(first_term, second_term), "offset": offset}
            )
    elif shot_type == "same_fn":
        for _ in range(pool_size):
            offset = random.randint(0, 10)
            shot_pool.append({"fn": base_fn["fn"], "offset": offset})
    elif shot_type == "ambigious":
        while len(shot_pool) < pool_size:
            fn_item = random.choice(list(ambiguous_sequences.items()))
            fns = np.random.choice(fn_item[1], 2, replace=False)
            shot_pool.extend(fns)

    shots = np.random.choice(shot_pool, size=n_shots, replace=False)
    # continue to draw if fn_item in shots
    while base_fn in shots:
        shots = np.random.choice(shot_pool, size=n_shots, replace=False)
    return shot_pool


def _cot(fn_item: dict, steps: int) -> str:
    """
    Create a chain of thought steps by resolving the function a step at a time
    for `:steps` steps.
    """
    prompt = COT_PROMPT
    for step in range(steps):
        completion = _resolve_fn(fn_item, step)
        prompt += COT_STEP.format(
            step, fn_item["fn"], step + fn_item["offset"], completion
        )
    return prompt


def _resolve_fn(fn_item: dict, step: int) -> int:
    # resolve function to a given completion
    # TODO: maybe move offset outside of here to make it clearer
    step = step + fn_item["offset"]
    return eval(fn_item["fn"])(step)


def _create_sequence_prompt(
    sequence: str,
    fn_item: dict,
    prompt_type: Literal["completion", "explanation"],
    use_cot=False,
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
    completion = ""
    if prompt_type == "completion":
        prompt += BASE_PROMPT_COMPLETION
        last_step = len(sequence.split(","))
        completion = _resolve_fn(fn_item, last_step)
    elif prompt_type == "explanation":
        prompt += BASE_PROMPT_EXPLANATION
        completion = fn_item["fn"]

    cot = ""
    if use_cot:
        cot = _cot(fn_item, len(sequence.split(",")) + 1)
    return prompt, completion, cot


def _sample_shots(
    sequence: str,
    fn_item: dict,
    n_shots: int,
    prompt_type: Literal["completion", "explanation"] = "completion",
    use_cot: bool = False,
    shot_type: Literal[
        "random", "same_fn", "same_class", "ambigious", "exclude_class"
    ] = "few_shot",
    ambiguous_sequences: dict = None,
):
    """
    Sample `:n_shots` number of shots.
    Initially we randomly generate `:_generate_shot_pool` the shots.
    """
    shots = _generate_shot_pool(
        n_shots=n_shots,
        base_fn=fn_item,
        shot_type=shot_type,
        ambiguous_sequences=ambiguous_sequences,
    )
    # for all the shots create sequence prompts
    prompts = []
    for shot in shots:
        # TODO: make this magic string more obvious
        steps = random.randint(2, 4)
        sequence = ",".join([str(_resolve_fn(shot, step)) for step in range(steps)])
        prompt, completion, cot = _create_sequence_prompt(
            sequence, shot, prompt_type, use_cot=use_cot
        )
        prompts.append({"prompt": prompt, "completion": completion, "cot": cot})

    return prompts


def generate_sequence_completion_prompt(
    sequence: str,
    fn_item: dict,
    prompt_type: Literal["completion", "explanation"] = "completion",
    use_cot: bool = False,
    n_shots: int = 0,
    shot_type: Literal[
        "random", "same_fn", "same_class", "ambigious", "exclude_class"
    ] = "random",
    ambiguous_sequences: dict = None,
) -> dict:
    """
    Generate sequence completion prompts
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

    shots = _sample_shots(
        sequence,
        fn_item,
        prompt_type=prompt_type,
        n_shots=n_shots,
        use_cot=use_cot,
        shot_type=shot_type,
        ambiguous_sequences=ambiguous_sequences,
    )

    for shot in shots:
        completion = str(shot["completion"])
        if use_cot:
            completion += shot["cot"]
        turns = [
            {"role": "user", "content": shot["prompt"]},
            {"role": "assistant", "content": completion},
        ]
        prompt_turns.extend(turns)

    prompt, completion, _ = _create_sequence_prompt(sequence, fn_item, prompt_type)

    prompt_turns.append({"role": "user", "content": prompt})
    return {"prompt_turns": prompt_turns, "completion": completion}
