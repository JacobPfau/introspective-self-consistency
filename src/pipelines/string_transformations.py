"""
This file generates ambigious string transformation tasks.

Example #1
#@#@ => @##@
@##@ =>
Possible transformations:
swap_one_left
swap_one_right


Example #2
#@#@ => @#@#
@#@# =>

Example #3
#@#@ => @#@@
@#@@ =>

"""
from typing import Literal, Tuple
import numpy as np

import random

SYSTEM_PROMPT = """
You are a puzzle assistant.
You are helping with string transformation puzzles.
The string transformations work step by step by applying a transformation to a sequence of characters.
You answer accurately and concisely.
"""

SYSTEM_PROMPT_COMPLETION = """
You only respond with the transformed sequence.
"""

SYSTEM_PROMPT_EXPLANATION = """
You only respond with code that describes the transformed sequence.
"""

BASE_PROMPT = """
For the transformation {transformation}
"""

BASE_PROMPT_COMPLETION = """
Complete the transformation using {next_string}
{next_string} =>
"""

BASE_PROMPT_EXPLANATION = """
Give the code that describes the transformation.
"""

invert = "lambda x: x[::-1]"
shift_left_n = "lambda x: x[{n}:] + x[:{n}]"
shift_right_n = "lambda x: x[-{n}:] + x[:-{n}]"
move_char_at_n_to_left = 'lambda x: x[:{n}] + x[{n}:].replace({char}, "", 1) + {char}'
move_char_at_n_to_right = 'lambda x: x[:{n}] + {char} + x[{n}:].replace({char}, "", 1)'
swap = "lambda x: x.translate(str.maketrans({char_1} + {char_2}, {char_2} + {char_1}))"
replace_count_n = "lambda x: x.replace({char_1}, {char_2}, {n})"

transformation_fns = {
    "replace_count_n": replace_count_n,
    "invert": invert,
    "shift_left_n": shift_left_n,
    "shift_right_n": shift_right_n,
    "swap": swap,
    "move_char_at_n_to_left": move_char_at_n_to_left,
    "move_char_at_n_to_right": move_char_at_n_to_right
}


# generate all possible combinations of two character sequences of fixed length n
def generate_all_two_char_sequences(char_1, char_2, n):
    all_two_char_sequences = [char_1 * n, char_2 * n]
    for i in range(n):
        for j in range(n):
            if i == j or i > j:
                continue
            all_two_char_sequences.append(char_1 * i + char_2 + char_1 * (n - i - 1))
            all_two_char_sequences.append(char_2 * i + char_1 + char_2 * (n - i - 1))
    return all_two_char_sequences


# rep_apply(func, n) returns a function that applies func n times
def rep_apply(func, n):
    if n > 1:
        rec_func = rep_apply(func, n - 1)
        return lambda x: func(rec_func(x))
    return func


def find_ambiguous_string_transformations(char_1, char_2, n):
    application_steps = 4
    n_variations = 4

    base_sequences = generate_all_two_char_sequences(char_1, char_2, n)
    transformations = generate_all_transformations(
        char_1, char_2, application_steps, n_variations, base_sequences
    )

    # a transformation is ambigious
    # if transformation_1['transformation'][i] == transformation_2['transformation'][i]
    # but transformation_1['transformation'][i + 1] != transformation_2['transformation'][i + 1]
    # ambigious sequence are then key'd by the sequence before and after the first transformation
    ambiguous_sequences = {}
    for transformation_1 in transformations:
        for transformation_2 in transformations:
            # if its the same transformation or not the same starting sequence, skip
            if (
                transformation_1 == transformation_2
                or transformation_1["sequence"] != transformation_2["sequence"]
            ):
                continue
            # offset maybe? for i in range(len(transformation_1['transformations']) - 1):
            if (
                transformation_1["transformations"][0]
                == transformation_2["transformations"][0]
                and transformation_1["transformations"][0 + 1]
                != transformation_2["transformations"][0 + 1]
            ):
                seq_key = (
                    transformation_1["sequence"]
                    + " => "
                    + transformation_1["transformations"][0]
                )
                if seq_key not in ambiguous_sequences:
                    ambiguous_sequences[seq_key] = []
                if (
                    transformation_1["transformation"]
                    not in ambiguous_sequences[seq_key]
                ):
                    ambiguous_sequences[seq_key].append(
                        transformation_1["transformation"]
                    )
                if (
                    transformation_2["transformation"]
                    not in ambiguous_sequences[seq_key]
                ):
                    ambiguous_sequences[seq_key].append(
                        transformation_2["transformation"]
                    )
    return ambiguous_sequences


def generate_all_transformations(
    char_1, char_2, application_steps, n_variations, base_sequences
):
    sequence_data = []
    for sequence in base_sequences:
        for transformation, transformation_fn in transformation_fns.items():
            if transformation == "invert":
                fn = transformation_fn
                eval_fn = eval(fn)
                sequence_data.append(
                    {
                        "sequence": sequence,
                        # apply the transformation n times
                        "transformation": fn,
                        "transformations": [
                            rep_apply(eval_fn, step + 1)(sequence)
                            for step in range(application_steps)
                        ],
                    }
                )
            if transformation == "swap":
                # swap one way
                fn = transformation_fn.format(
                    char_1=f"'{char_1}'", char_2=f"'{char_2}'"
                )
                eval_fn = eval(fn)
                sequence_data.append(
                    {
                        "sequence": sequence,
                        # apply the transformation n times
                        "transformation": fn,
                        "transformations": [
                            rep_apply(eval_fn, step + 1)(sequence)
                            for step in range(application_steps)
                        ],
                    }
                )
                # swap the other way
                fn = transformation_fn .format(
                    char_1=f"'{char_1}'", char_2=f"'{char_2}'"
                )
                eval_fn = eval(fn)
                sequence_data.append(
                    {
                        "sequence": sequence,
                        # apply the transformation n times
                        "transformation": fn,
                        "transformations": [
                            rep_apply(eval_fn, step + 1)(sequence)
                            for step in range(application_steps)
                        ],
                    }
                )
            if transformation in ["shift_left_n", "shift_right_n"]:
                for i in range(n_variations):
                    fn = transformation_fn .format(n=i)
                    eval_fn = eval(fn)
                    sequence_data.append(
                        {
                            "sequence": sequence,
                            # apply the transformation n times
                            "transformation": fn,
                            "transformations": [
                                rep_apply(eval_fn, step + 1)(sequence)
                                for step in range(application_steps)
                            ],
                        }
                    )
            if transformation in [
                "move_char_at_n_to_left",
                "move_char_at_n_to_right",
            ]:
                # move char_1
                for i in range(n_variations):
                    fn = transformation_fn .format(n=i, char=f"'{char_1}'")
                    eval_fn = eval(fn)
                    sequence_data.append(
                        {
                            "sequence": sequence,
                            # apply the transformation n times
                            "transformation": fn,
                            "transformations": [
                                rep_apply(eval_fn, step + 1)(sequence)
                                for step in range(application_steps)
                            ],
                        }
                    )
                # move char_2
                for i in range(n_variations):
                    fn = transformation_fn .format(n=i, char=f"'{char_2}'")
                    eval_fn = eval(fn)
                    sequence_data.append(
                        {
                            "sequence": sequence,
                            # apply the transformation n times
                            "transformation": fn,
                            "transformations": [
                                rep_apply(eval_fn, step + 1)(sequence)
                                for step in range(application_steps)
                            ],
                        }
                    )
            if transformation == "replace_count_n":
                # replace char_1 with char_2
                for i in range(n_variations):

                    fn = transformation_fn .format(
                        char_1=f"'{char_1}'", char_2=f"'{char_2}'", n=i
                    )
                    eval_fn = eval(fn)
                    sequence_data.append(
                        {
                            "sequence": sequence,
                            # apply the transformation n times
                            "transformation": fn,
                            "transformations": [
                                rep_apply(eval_fn, step + 1)(sequence)
                                for step in range(application_steps)
                            ],
                        }
                    )
                # replace char_2 with char_1
                for i in range(n_variations):
                    fn = transformation_fn .format(
                        char_1=f"'{char_2}'", char_2=f"'{char_1}'", n=i
                    )
                    eval_fn = eval(fn)
                    sequence_data.append(
                        {
                            "sequence": sequence,
                            # apply the transformation n times
                            "transformation": fn,
                            "transformations": [
                                rep_apply(eval_fn, step + 1)(sequence)
                                for step in range(application_steps)
                            ],
                        }
                    )
    return sequence_data


def _create_string_transformation_prompt(
    sequence: str,
    fn_item: dict,
    prompt_type: Literal["completion", "explanation"],
    use_cot=False,
) -> Tuple[str, str, str]:
    prompt = BASE_PROMPT.format(transformation=sequence)
    completion = ""
    if prompt_type == "completion":
        prompt += BASE_PROMPT_COMPLETION.format(
            next_string=sequence.split('=>')[1].strip()
        )
        completion = eval(fn_item["fn"])(sequence.split('=>')[1].strip())
    elif prompt_type == "explanation":
        prompt += BASE_PROMPT_EXPLANATION
        completion = fn_item["fn"]

    cot = ""
    if use_cot:
        raise NotImplementedError("Chain of thought not implemented yet")
        # cot = _cot(fn_item, len(sequence.split(",")) + 1)
    return prompt, completion, cot


def _generate_shot_pool(
    sequence_length: int,
    char_1: int,
    char_2: int,
    pool_size: int = 10
):
    fn_pool = list(transformation_fns.items())
    shot_pool = []
    # we generate a prompt_pool with random parameters
    # TODO: move these magic strings to somewhere more visible
    base_sequences = generate_all_two_char_sequences(char_1, char_2, sequence_length)
    for _ in range(pool_size):
        transformation_fn = random.choice(fn_pool)
        selected_position = random.randint(0, sequence_length)
        char_1 = random.choice([char_1, char_2])
        char_2 = random.choice([char_1, char_2])
        sequence = random.choice(base_sequences)

        if transformation_fn[0] == "invert":
            fn = transformation_fn[1]
            eval_fn = eval(fn)

            shot_pool.append(
                {"fn": fn, "sequence": sequence, "transformations": [
                    eval_fn(sequence)], "transformation": "{} => {}".format(sequence, eval_fn(sequence))}
            )
        elif transformation_fn[0] == "swap":
            fn = transformation_fn[1].format(
                char_1=f"'{char_1}'", char_2=f"'{char_2}'"
            )
            eval_fn = eval(fn)

            shot_pool.append({"fn": fn, "sequence": sequence, "transformations": [eval_fn(
                sequence)], "transformation": "{} => {}".format(sequence, eval_fn(sequence))})
        if transformation_fn[0] in ["shift_left_n", "shift_right_n"]:
            fn = transformation_fn[1].format(n=selected_position)
            eval_fn = eval(fn)

            shot_pool.append({"fn": fn, "sequence": sequence, "transformations": [eval_fn(
                sequence)], "transformation": "{} => {}".format(sequence, eval_fn(sequence))})
        if transformation_fn[0] in [
            "move_char_at_n_to_left",
            "move_char_at_n_to_right",
        ]:
            fn = transformation_fn[1].format(n=selected_position, char=f"'{char_1}'")
            eval_fn = eval(fn)
            shot_pool.append({"fn": fn, "sequence": sequence, "transformations": [eval_fn(
                sequence)], "transformation": "{} => {}".format(sequence, eval_fn(sequence))})
        if transformation_fn[0] in ["replace_count_n"]:
            fn = transformation_fn[1].format(
                char_1=f"'{char_1}'", char_2=f"'{char_2}'", n=selected_position
            )
            eval_fn = eval(fn)
            shot_pool.append({"fn": fn, "sequence": sequence, "transformations": [eval_fn(
                sequence)], "transformation": "{} => {}".format(sequence, eval_fn(sequence))})

    return shot_pool


def _sample_shots(
    fn_item: dict,
    sequence_length: int,
    char_1: str,
    char_2: str,
    n_shots: int = 8,
    prompt_type: Literal["completion", "explanation"] = "completion",
    use_cot: bool = False,
):
    """
    Sample `:n_shots` number of shots.
    Initially we randomly generate `:_generate_shot_pool` the shots.
    """
    # TODO: implement "oracle", "adversarial", "ambigious" few shot strategies
    shot_pool = _generate_shot_pool(sequence_length, char_1, char_2, pool_size=n_shots * 2)
    shots = np.random.choice(shot_pool, size=n_shots, replace=False)
    # continue to draw if fn_item in shots
    while fn_item in shots:
        shots = np.random.choice(shot_pool, size=n_shots, replace=False)

    # for all the shots create sequence prompts
    prompts = []
    for shot in shots:
        # TODO: make this magic string more obvious
        prompt, completion, cot = _create_string_transformation_prompt(
            shot['transformation'], shot, prompt_type, use_cot=use_cot
        )
        prompts.append({"prompt": prompt, "completion": completion, "cot": cot})

    return prompts


def generate_string_transformation_prompt(
    sequence: str,
    fn_item: dict,
    sequence_length: str,
    char_1: str,
    char_2: str,
    prompt_type: Literal["completion", "explanation"] = "completion",
    use_cot: bool = False,
    n_shots: int = 0,
) -> dict:
    """
    Generate string transformation prompts
    including support for few_shot with `:n_shots`
    and chain of thought step completions with `:use_cot`

    Returns:
        dict:
    """

    if use_cot:
        raise NotImplementedError("Chain of thought not implemented yet")

    # TODO: this should be generic so it isn't coupled to ChatGPT
    prompt_turns = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
    ]

    shots = _sample_shots(
        fn_item, sequence_length, char_1, char_2, prompt_type=prompt_type, n_shots=n_shots, use_cot=use_cot
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

    prompt, completion, _ = _create_string_transformation_prompt(sequence, fn_item, prompt_type)

    prompt_turns.append({"role": "user", "content": prompt})
    return {"prompt_turns": prompt_turns, "completion": completion}
