"""
This file contains functions for generating base-b ambiguous sequence completition problems.

Differences from integer sequence completion:
- Uses base-b sequences
- sequence functions does not include indexing
- Sequences are compared as strings i.e. all sequence elements are concatenated without separators into a string
- Offsets are handled differently.
- Always disambiguates
"""

from random import shuffle
from typing import Tuple

# Same as integer sequence functions but excluding indexing
default_sequence_functions = {
    "arithmetic_progression": "lambda x: ({} * x) + {}",
    "geometric_progression": "lambda x: ({} * x) * {}",
    "exponential_progression": "lambda x: ({} * x) ** {}",
    "power_progression": "lambda x: {} ** ({} * x)",
    # "bit_or_progression": "lambda x: ({} * x) | {}",
    # "modular_progression": "lambda x: (x * {}) % ({}+1)",
    "recursive_progression": (
        "(lambda a:lambda v:a(a,v))(lambda fn,x:1 if x==0 else {} * x * fn(fn,x-1) + {})"
    ),
}


def numberToBase(n, b):
    if n == 0:
        return "0"
    digits = []
    while n:
        digits.append(str(n % b))
        n //= b
    return "".join(digits[::-1])


def find_ambiguous_string_sequences(
    base: int = 2,
    index: int = 0,
    max_constant_term_one: int = 5,
    max_constant_term_two: int = 5,
    len_to_check: int = 8,
    disambiguate_len: int = 8,
    sequence_functions: dict = default_sequence_functions,
) -> dict:
    """
    Find ambiguous_integer_sequences using brute force search
    over a set of progressions.

    A sequence is said to ambiguous if an initial set of two or more completions
    can be generated by different functions.

    Args:
        disambiguate_len: The number of characters to check for disambiguation after finding an ambiguous pair.

    Returns a dict of ambiguous sequence objects:
    - key: ambiguous_sequence
    - value: list of function_candidates with their offsets.
    """
    progression_base_fns = sequence_functions
    progressions_to_check = []
    for const_term_one in range(max_constant_term_one + 1):
        for const_term_two in range(max_constant_term_two + 1):
            for name, progression in progression_base_fns.items():
                progressions_to_check.append(
                    (
                        (name, const_term_one, const_term_two),
                        progression.format(const_term_one + 1, const_term_two),
                    )
                )
    ambiguous_sequences = {}
    for ind, pair in enumerate(progressions_to_check):
        metadata_a, fn_a = pair
        for metadata_b, fn_b in progressions_to_check[ind + 1 :]:
            if metadata_a[0] == metadata_b[0]:
                continue

            # check the sequence progressions
            # through n steps and add to ambiguous_sequences if ambiguous
            check_ambiguity(
                base,
                index,
                len_to_check,
                ambiguous_sequences,
                fn_a,
                fn_b,
                metadata_a,
                metadata_b,
                disambiguate_len,
            )
    return ambiguous_sequences


def check_ambiguity(
    base: int,
    index: int,
    ambig_check_len: int,
    ambiguous_sequences: dict,
    fn_a: str,
    fn_b: str,
    metadata_a: Tuple[str, int, int],
    metadata_b: Tuple[str, int, int],
    disambiguate_len: int,
) -> None:
    """
    Check the ambiguity of two sequence generating function

    We step through ambig_check_len.
    As we do we check if fn_a(step) == fn_b(step) and
    fn_a(step + 1) != fn_b(step + 1)

    We account for unaligned sequences by iterating through offsets.

    !This function mutates ambiguous_sequences and adds ambiguous sequences
    as it goes.
    """

    str_a, str_b = "", ""
    ambig_check_ind = ambig_check_len + index
    total_len = disambiguate_len + ambig_check_ind
    for step in range(index, total_len):
        fn_a_step = eval(fn_a)(step)
        fn_b_step = eval(fn_b)(step)

        str_a += numberToBase(fn_a_step, base)
        str_b += numberToBase(fn_b_step, base)
        if len(str_a) >= total_len and len(str_b) >= total_len:
            break
    if (
        str_a[:ambig_check_len] != str_b[:ambig_check_len]
    ):  # if the sequences are not the same, they are not ambiguous
        return
    if (
        str_a[:total_len] == str_b[:total_len]
    ):  # if the sequences are the same, they are not disambiguate-able
        return
    ambig_substring = str_a[:ambig_check_len]
    disambiguating_ind = None
    for i, char in enumerate(str_a[ambig_check_len:]):
        if char != str_b[ambig_check_len + i]:
            disambiguating_ind = ambig_check_len + i
            break

    if ambig_substring not in ambiguous_sequences:
        ambiguous_sequences[ambig_substring] = []
    ambiguous_sequences[ambig_substring].append(
        {
            "data": fn_a,
            "metadata": metadata_a,
            "disambiguating_pair_data": fn_b,
            "disambiguating_pair_metadata": metadata_b,
            "disambiguating_step": disambiguating_ind,
        }
    )


def return_ambiguity(
    base: int,
    index: int,
    ambig_check_len: int,
    fn_a: str,
    fn_b: str,
) -> None:
    """
    Check the ambiguity of two sequence generating function
    Returns True if ambiguous, False otherwise
    """

    str_a, str_b = "", ""
    ambig_check_ind = ambig_check_len + index
    for step in range(index, ambig_check_ind):
        fn_a_step = eval(fn_a)(step)
        fn_b_step = eval(fn_b)(step)

        str_a += numberToBase(fn_a_step, base)
        str_b += numberToBase(fn_b_step, base)
        if len(str_a) >= ambig_check_ind and len(str_b) >= ambig_check_ind:
            break
    if (
        str_a[:ambig_check_len] != str_b[:ambig_check_len]
    ):  # if the sequences are not the same, they are not ambiguous
        return False
    else:
        return True


def find_unambiguous_string_sequences(
    base: int = 2,
    index: int = 0,
    max_constant_term_one: int = 5,
    max_constant_term_two: int = 5,
    len_to_check: int = 8,
    sequence_functions: dict = default_sequence_functions,
) -> dict:
    """
    Find ambiguous_integer_sequences using brute force search
    over a set of progressions.

    A sequence is said to ambiguous if an initial set of two or more completions
    can be generated by different functions.

    Args:
        disambiguate_len: The number of characters to check for disambiguation after finding an ambiguous pair.

    Returns a dict of ambiguous sequence objects:
    - key: ambiguous_sequence
    - value: list of function_candidates with their offsets.
    """
    progression_base_fns = sequence_functions
    progressions_to_check = []
    unambiguous_sequences = set()
    for const_term_one in range(max_constant_term_one + 1):
        for const_term_two in range(max_constant_term_two + 1):
            for name, progression in progression_base_fns.items():
                fn = progression.format(const_term_one + 1, const_term_two)
                progressions_to_check.append((name, fn))
                unambiguous_sequences.add((name, fn))

    for ind, pair_a in enumerate(progressions_to_check):
        name_a, fn_a = pair_a
        for pair_b in progressions_to_check[ind + 1 :]:
            name_b, fn_b = pair_b
            if return_ambiguity(
                base,
                index,
                len_to_check,
                fn_a,
                fn_b,
            ):
                unambiguous_sequences.discard(pair_a)
                unambiguous_sequences.discard(pair_b)
    return unambiguous_sequences


def main():
    ambiDict = find_ambiguous_string_sequences(
        base=2,
        index=1,
        max_constant_term_one=7,
        max_constant_term_two=7,
        len_to_check=8,
        disambiguate_len=8,
        sequence_functions=default_sequence_functions,
    )
    occurences = {fn: [] for fn in default_sequence_functions.keys()}
    for value in ambiDict.values():
        for occ in occurences.items():
            occ[1].append(0)
    for v, value in enumerate(ambiDict.values()):
        for d in value:
            type = d["metadata"][0]
            if occurences[type][v] == 0:
                occurences[type][v] = 1

    print(len(ambiDict.keys()))
    print(ambiDict.keys())
    for k, v in occurences.items():
        print(k, "\n", sum(v), v)

    ind = 1
    unique = find_unambiguous_string_sequences(
        index=ind,
        max_constant_term_one=7,
        max_constant_term_two=7,
        sequence_functions=default_sequence_functions,
    )
    print(len(unique))
    unique = list(unique)
    shuffle(unique)
    seqs = []
    for pair in unique:
        binseq = ""
        step = ind
        while len(binseq) < 8:
            binseq += numberToBase(eval(pair[1])(step), 2)
            step += 1
        if step >= 4:
            if len(binseq) < 9:
                binseq += numberToBase(eval(pair[1])(step), 2)
            seqs.append((pair[0], pair[1], binseq[:8], binseq[:9]))
            print(seqs[-1])


if __name__ == "__main__":
    main()
