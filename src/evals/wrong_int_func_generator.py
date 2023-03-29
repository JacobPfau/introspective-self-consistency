import random
from typing import List, Tuple, Union

from pipelines.sequence_completions import sequence_functions


def generate_wrong_functions(
    sequence: Union[str, List[int]],
    num_functions: int = 5,
    offset_range: Tuple[int, int] = (0, 10),
    num_range: Tuple[int, int] = (0, 10),
    func_pool: dict = sequence_functions,
) -> List[str]:
    """
    Given an integer sequence, and a method of generating functions, generate a list of incorrect functions.
    Uses the sequence_functions dictionary in pipelines.sequence_completions, with offsets
    """
    if isinstance(sequence, str):
        # Turn the sequence into a list of ints
        sequence = [int(x.strip()) for x in sequence.split(",")]
    sequence_length = len(sequence)
    output = []
    i = 0
    while i < len(range(num_functions)):
        fn, _ = _generate_random_function(func_pool, num_range, offset_range)
        # TODO: will just check no equivalence for the first ten possible offsets, might want to change this
        correct = False
        for offset in range(10):
            # Check that the candidate is incorrect
            for step in range(sequence_length):
                fn_step = eval(fn)(step + offset)
                if fn_step != sequence[step]:
                    break
                elif step == sequence_length - 1:
                    correct = True
                    break
            if correct:
                break
        if not correct:
            i += 1
            output.append(fn)

    return output


def _generate_random_function(
    func_pool: dict, num_range: Tuple[int, int], offset_range: Tuple[int, int]
) -> Tuple[str, int]:
    """
    Given a pool of functions, randomly sample one, and an offset.
    """
    fn = random.choice(list(func_pool.values()))
    # Incorporate two numbers into the function
    # TODO: refactor this, is kind of ugly
    fn = fn.format(
        random.choice(list(range(*num_range))), random.choice(list(range(*num_range)))
    )
    offset = random.choice(list(range(offset_range[0], offset_range[1])))
    return (fn, offset)


if __name__ == "__main__":
    sequence = "1, 2, 3, 4, 5"
    output = generate_wrong_functions(sequence)
    print(output)
