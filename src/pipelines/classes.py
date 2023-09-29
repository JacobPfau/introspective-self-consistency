from enum import Enum


class ShotSamplingType(Enum):
    """
    Sampling method for few shot examples
    """

    RANDOM = "random"
    SAME_FN = "same_fn"
    SAME_CLASS = "same_class"
    AMBIGUOUS = "ambiguous"
    EXCLUDE_CLASS = "exclude_class"


class TaskType(Enum):
    COMPLETION = "completion"
    EXPLANATION = "explanation"
