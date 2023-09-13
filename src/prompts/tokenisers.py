"""
Different methods for formatting numbers in the prompt.
Potentially necessary to get around issues with tokenisers.
"""


def period_space_format(number: str) -> str:
    """
    Take a number in an arbitrary base, and change how it's displayed.
    This is to help reduce issues with tokenisers
    For now, take e.g. 01010 -> 0. 1. 0. 1. 0. ,
    """
    return ". ".join(list(number)) + ". "


number_format_dict = {
    "None": lambda x: x,
    "PeriodSpace": period_space_format,
}
