from typing import Dict, List, Union


def extend_prompt(
    prompt_text: Union[str, List[Dict[str, str]]],
    shot_text: Union[str, List[Dict[str, str]]],
) -> Union[str, List[Dict[str, str]]]:
    """
    Given a prompt, extend it with the shot text.
    """
    if isinstance(prompt_text, str):
        assert isinstance(shot_text, str)
        return shot_text + prompt_text
    elif isinstance(prompt_text, list):
        assert isinstance(shot_text, list)
        return shot_text + prompt_text
    else:
        raise ValueError(f"Invalid prompt type: {type(prompt_text)}")
