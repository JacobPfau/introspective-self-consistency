import functools
import logging


def log_exceptions(logger: logging.Logger):
    """
    Decorator to catch and log exceptions.

    Useful in combination with hydra to make sure that also uncaught exceptions are properly logged to file.
    """

    def decorator(func):
        @functools.wraps(func)
        def decorated(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(e)
                raise e

        return decorated

    return decorator


def reformat_self_consistency_results(results):
    """
    Add in the total number of samples and the percentage of consistent
    and inconsistent explanations, alongside invalid explanations. Calculate
    this average across all sequences, and add it to the results dict.
    """
    consistent, inconsistent, invalid = 0, 0, 0
    for sequence in results:
        consistent += results[sequence]["consistent"]
        inconsistent += results[sequence]["inconsistent"]
        invalid += results[sequence]["invalid"]

    total = consistent + inconsistent + invalid
    consistent_percentage = consistent / total
    inconsistent_percentage = inconsistent / total
    invalid_percentage = invalid / total

    results["total"] = total
    results["consistent"] = consistent_percentage
    results["inconsistent"] = inconsistent_percentage
    results["invalid"] = invalid_percentage

    return results
