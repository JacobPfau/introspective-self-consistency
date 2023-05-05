from models.openai_model import CHAT_MODEL_NAME  # , DAVINCI_MODEL_NAME
from q11.prompts.continuation_prompt import create_continuation_prompt


def test_completion_prompt():
    sequence = [1, 2, 3, 4, 5]
    distribution = "default"
    model_name = CHAT_MODEL_NAME
    base = 10
    prompt = create_continuation_prompt(
        sequence=sequence,
        distribution=distribution,
        model_name=model_name,
        base=base,
    )
    print(prompt)
    print("buggo")


if __name__ == "__main__":
    test_completion_prompt()
