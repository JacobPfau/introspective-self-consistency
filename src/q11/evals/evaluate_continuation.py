def valid_continuation(
    model_continuation: str,
) -> bool:
    """
    Given a continuation as supplied by the model,
    return whether it is a valid integer or not.
    """
    try:
        # TODO: Work for arbitrary base continuation
        int(model_continuation)
    except ValueError:
        return False
    else:
        return True
