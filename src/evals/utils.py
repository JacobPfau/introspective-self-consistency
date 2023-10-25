import random
from typing import Tuple


def generate_random_function(
    func_pool: dict, num_range: Tuple[int, int], offset_range: Tuple[int, int], seed=0
) -> Tuple[str, int]:
    """
    Given a pool of functions, randomly sample one, and an offset.
    """
    if seed != 0:
        random.seed(seed)
    fn: str = random.choice(list(func_pool.values()))
    # Incorporate two numbers into the function
    # TODO: refactor this, is kind of ugly
    fn = fn.format(
        random.choice(list(range(*num_range))), random.choice(list(range(*num_range)))
    )
    offset = random.choice(list(range(offset_range[0], offset_range[1])))
    return (fn, offset)


def reformat_function(fn: str, offset: int, base: int = 10) -> str:
    """
    Reformat a function to incorporate an offset, so the function is zero indexed.
    """
    first_occurrence = fn.find("x")
    replacement = f"(x + {offset})"
    if first_occurrence != -1:
        fn = fn[:first_occurrence] + "<placeholder>" + fn[first_occurrence + len("x") :]

    # replace all occurrences of x
    fn = fn.replace("x", replacement)
    # restore the first occurrence
    fn = fn.replace("<placeholder>", "x", 1)

    if base == 2:
        if "fn" in fn:
            # If the function is recursive, need to handle it differently
            # Find a(a,v)
            first_occurrence = fn.find("a(a,v)")
            # replace a(a,v) with bin(a(a,v))
            fn = (
                fn[:first_occurrence]
                + "bin(a(a,v))"
                + fn[first_occurrence + len("a(a,v)") :]
            )

        else:
            # Wrap the output in a binary conversion
            prefix, suffix = fn.split(":", 1)
            # Add bin around the calculation part and join back together
            fn = prefix + ": bin(" + suffix.strip() + ")"

    return fn
