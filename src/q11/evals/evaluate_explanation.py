from typing import List


def valid_explanation(
    fn_form: str,
    offset: int,
    sequence_length: int,
) -> bool:
    """
    Given a function form and an offset as supplied by the model,
    return whether the string is a valid python function.
    """
    try:
        # TODO: need to have this work for an arbitrary number of arguments
        [eval(fn_form.format(i + offset)) for i in range(sequence_length)]
    except SyntaxError:
        return False
    except NameError:
        return False
    except TypeError:
        return False
    except ValueError:
        return False
    else:
        return True


def correct_explanation(
    fn_form: str,
    offset: int,
    sequence_length: int,
    sequence: List[int],
) -> bool:
    """
    Given a function form and an offset as supplied by the model,
    return whether the function correctly generates the sequence.
    """
    return all(
        eval(fn_form.format(i + offset)) == sequence[i] for i in range(sequence_length)
    )


def generate_continuation(
    fn_form: str,
    offset: int,
    sequence_length: int,
) -> List[int]:
    """
    Given a function form and an offset as supplied by the model,
    generate the next element of the sequence.
    """
    return eval(fn_form.format(sequence_length + offset + 1))
