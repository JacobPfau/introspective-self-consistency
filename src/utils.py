import functools
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import git  # installed with `pip install gitpython`
from hydra.experimental.callback import Callback
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@contextmanager
def in_subdir(path: str):
    """
    Switch into a subdirectory and switch back when leaving the context.
    """
    origin = Path.cwd().absolute()
    subdir = origin / path
    try:
        subdir.mkdir(parents=True, exist_ok=True)
        os.chdir(subdir)
        yield
    finally:
        os.chdir(origin)


def auto_subdir(func):
    """
    Switch automatically into a subdirectory named after the function name.

    Useful in combination with hydras way of logging and storing results.
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        subdir = Path.cwd() / func.__name__
        with in_subdir(subdir):
            logger.info(f"Changed directory to {subdir}")
            return func(*args, **kwargs)

    return wrapped


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


class LogGitHashCallback(Callback):
    """
    LogGitHashCallback logs, on the start of every run, the git hash of the current commit and changed files (if any).

    To use it include the following in your config:
        ```yaml
        hydra:
          callbacks:
            git_logging:
              _target_: callbacks.hydra.LogGitHashCallback
        ```

    (adapted from https://stackoverflow.com/a/74133166)
    """

    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        _log_git_sha()


def _log_git_sha():
    log = logging.getLogger(__name__)

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    log.info(f"Git sha: {sha}")

    changed_files = [item.a_path for item in repo.index.diff(None)]
    if changed_files:
        log.info(f"Changed files: {changed_files}")

    diff = repo.git.diff()
    if diff:
        log.info(f"Git diff:\n{diff}")


def reformat_self_consistency_results(results):
    """
    Add in the total number of samples and the percentage of consistent
    and inconsistent explanations, alongside invalid explanations. Calculate
    this average across all sequences, and add it to the results dict.
    """
    consistent, inconsistent, incorrect, invalid = 0, 0, 0, 0
    for sequence in results:
        consistent += results[sequence]["consistent"]
        inconsistent += results[sequence]["inconsistent"]
        incorrect += results[sequence]["incorrect"]
        invalid += results[sequence]["invalid"]

    total = consistent + inconsistent + invalid + incorrect
    consistent_percentage = consistent / total
    inconsistent_percentage = inconsistent / total
    incorrect_percentage = incorrect / total
    invalid_percentage = invalid / total


    results["total"] = total
    results["consistent"] = consistent_percentage
    results["inconsistent"] = inconsistent_percentage
    results["incorrect"] = incorrect_percentage
    results["invalid"] = invalid_percentage

    return results
