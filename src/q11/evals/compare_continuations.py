def compare_continuations(
    continuation: int,
    explanation: str,
    model_offset: int,
    sequence_length: int,
) -> bool:
    """
    Given a valid continuation and a valid explanation, return whether the explanation
    gives the same continuation as the model.
    """
    index = sequence_length + model_offset + 1
    explanation_continuation = eval(explanation.format(index))
    return continuation == explanation_continuation
