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

Possible string operations:


"""

invert = "lambda x: x[::-1]"
shift_left_n = "lambda x: x[{n}:] + x[:{n}]"
shift_right_n = "lambda x: x[-{n}:] + x[:-{n}]"
move_char_at_n_to_left = 'lambda x: x[:{n}] + x[{n}:].replace({char}, "", 1) + {char}'
move_char_at_n_to_right = 'lambda x: x[:{n}] + {char} + x[{n}:].replace({char}, "", 1)'
swap = "lambda x: x.translate(str.maketrans({char_1} + {char_2}, {char_2} + {char_1}))"
replace_count_n = "lambda x: x.replace({char_1}, {char_2}, {n})"

transformation_fns = [
    ("replace_count_n", replace_count_n),
    ("invert", invert),
    ("shift_left_n", shift_left_n),
    ("shift_right_n", shift_right_n),
    ("swap", swap),
    ("move_char_at_n_to_left", move_char_at_n_to_left),
    ("move_char_at_n_to_right", move_char_at_n_to_right),
]


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
        for transformation in transformation_fns:
            if transformation[0] == "invert":
                fn = transformation[1]
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
            if transformation[0] == "swap":
                # swap one way
                fn = transformation[1].format(
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
                fn = transformation[1].format(
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
            if transformation[0] in ["shift_left_n", "shift_right_n"]:
                for i in range(n_variations):
                    fn = transformation[1].format(n=i)
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
            if transformation[0] in [
                "move_char_at_n_to_left",
                "move_char_at_n_to_right",
            ]:
                # move char_1
                for i in range(n_variations):
                    fn = transformation[1].format(n=i, char=f"'{char_1}'")
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
                    fn = transformation[1].format(n=i, char=f"'{char_2}'")
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
            if transformation[0] == "replace_count_n":
                # replace char_1 with char_2
                for i in range(n_variations):

                    fn = transformation[1].format(
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
                    fn = transformation[1].format(
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
