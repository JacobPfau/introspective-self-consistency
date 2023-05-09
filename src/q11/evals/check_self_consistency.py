from typing import List, Union

from models.openai_model import CHAT_MODEL_NAME, DAVINCI_MODEL_NAME


def generate_prompt(
    num_shots: int,
    model_name: str,
    temperature: float,
    method: str,
) -> Union[str, List[dict]]:
    """
    Generate prompts which will be used to generate explanations and continuations.

    For (text) continuations, this will be of the form:

    Here are some examples of sequence continuations.
    Q: 2, 4, 6,
    A: 8

    Q: 1, 2, 3, 4, 5,
    A: 6

    Q: 9, 16, 25, 36
    A: 49

    ***CONTINUATION_PROMPT***

    A:

    For (text) explanations, this will be of the form:

    Here are some examples of sequence explanations, i.e. python functions
    which could have generated the preceding sequences, with associated offset.
    Sequence: 2, 4, 6
    Explanation: lambda x: 2*x
    Offset: 0

    Sequence: 1, 2, 3, 4, 5
    Explanation: lambda x: x
    Offset: 1

    Sequence: 9, 16, 25, 36
    Explanation: lambda x: x**2
    Offset: 3

    ***EXPLANATION_PROMPT***

    Explanation:


    """

    if model_name == DAVINCI_MODEL_NAME:
        explanation_prompt =
