import evals.utils as utils
from evals.prompts.choose_function import function_selection_prompt
from models.openai_model import CHAT_MODEL_NAME  # , DAVINCI_MODEL_NAME

# TODO: make these proper tests


def test_generate_wrong_functions():
    sequence = "1, 2, 3, 4, 5"
    output = utils.generate_wrong_functions(sequence)
    print(output)


def test_choose_function():
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
        try:
            result = utils.choose_function(
                possible_functions=possible_functions,
                correct_function_indices=correct_function_indices,
                target_sequence=target_sequence,
                prompt=prompt,
                model_name=model_name,
            )
        except ValueError as e:
            print(e)
        else:
            print(result)


def prompt_test():
    prompt = function_selection_prompt(
        num_shots=5,
        sequence_length=8,
        num_functions=6,
        use_cot=False,
        model_name=CHAT_MODEL_NAME,
    )
    print(prompt)
    print("buggo")


if __name__ == "__main__":
    prompt_test()
