from typing import List

from models.openai_model import generate_chat_completion, generate_completion

DAVINCI_MODEL_NAME = "text-davinci-003"


def format_question(
    prompt: str,
    target_sequence: str,
    functions: List[str],
) -> str:
    formatted_answers = "\n".join(
        [f"{i+1}. {func}" for i, func in enumerate(functions)]
    )
    result = f"{prompt}\n{target_sequence}\n{formatted_answers}\nA:"
    return result


def parse_model_response(model_response: str) -> int:
    """
    Parse the model's response to get the index of the function it chose.
    """
    model_response = model_response.strip()
    if model_response == "":
        return -1
    # If the answer isn't correctly formatted, report an error
    model_response = model_response.split("\n .,")
    if len(model_response) > 1:
        return -2
    else:
        model_response = model_response[0]
    # If the answer isn't a number, report an error
    try:
        model_response = int(model_response)
    except ValueError:
        return -3
    return model_response


def choose_function(
    possible_functions: List[str],
    correct_function_indices: List[int],
    target_sequence: str,
    prompt: str,
    model_name: str,
    temperature: float = 0.0,
) -> int:
    """
    Prompt a model to chose a function, from a list of possible functions, that generated a target sequence.
    The prompt should provide instructions for the task. We will edit the prompt to include the possible functions.
    Compare the model's choice to the correct function (which is also provided).
    Assume for now that the model is an openai model, either DAVINCI or CHAT.

    Returns 1 if the model's response is correct, 0 if the model's response is incorrect, and < 0 if the model's response
    is invalid (wrong format, or empty)
    """
    # First, format the prompt to include the possible functions
    formatted_prompt = format_question(
        prompt=prompt,
        target_sequence=target_sequence,
        functions=possible_functions,
    )
    print(formatted_prompt)
    if model_name == "DAVINCI":

        # Feed this into the model
        model_response = generate_completion(
            prompt=formatted_prompt,
            temperature=temperature,
            max_tokens=256,
            model=DAVINCI_MODEL_NAME,
        )
    elif model_name == "CHAT":
        # Feed this into the model
        model_response = generate_chat_completion(
            # TODO: make this more general, to include multiple turns
            # for few shot examples
            prompt_turns=[{"text": formatted_prompt}],
            temperature=temperature,
            max_tokens=256,
            model=model_name,
        )
    # Parse the model's response to get the index of the function it chose
    model_response = parse_model_response(model_response)
    print(model_response)
    print(correct_function_indices)

    # If the model's response is not a valid index, return an error
    if model_response < 0:
        return model_response
    # Compare the model's response to the correct function
    if model_response in correct_function_indices:
        return 1
    # If the model's response is incorrect, return 0
    else:
        return 0


if __name__ == "__main__":
    with open("evals/prompts/choose_function.txt") as f:
        prompt = f.read()
        possible_functions = [
            "lambda x: 2 * x",
            "lambda x: 3 ** (4 * x)",
            "lambda x: 2 ** x",
            "lambda x: 21",
        ]
        correct_function_indices = [3]
        target_sequence = "1,2,4,8"
        model_name = "DAVINCI"
        result = choose_function(
            possible_functions=possible_functions,
            correct_function_indices=correct_function_indices,
            target_sequence=target_sequence,
            prompt=prompt,
            model_name=model_name,
        )
        print(result)
