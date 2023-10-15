from src.evals.check_self_consistency import self_consistency_evaluation
from src.evals.prompts.continuation_prompt import create_continuation_prompt
from src.evals.prompts.explanation_prompt import create_explanation_prompt


def test_completion_prompt():
    sequence = [1, 2, 3, 4, 5]
    task_prompt = "self-consistency"
    model_name = "gpt-3.5-turbo-0301"
    base = 2
    prompt = create_continuation_prompt(
        sequence=sequence,
        task_prompt=task_prompt,
        model_name=model_name,
        base=base,
    )
    print(prompt)
    print("buggo")


def test_create_continuation_prompt():
    sequence = [1, 2, 3, 4, 5, 6]
    task_prompt = "max-probability"
    role_prompt = "gpt-og"
    model_name = "gpt-3.5-turbo-0301"
    base = 2
    shots = 2
    shot_method = "random"
    prompt = create_continuation_prompt(
        sequence=sequence,
        task_prompt=task_prompt,
        role_prompt=role_prompt,
        model_name=model_name,
        base=base,
        shots=shots,
        shot_method=shot_method,
        seed=21,
    )
    print(prompt)


def test_create_explanation_prompt():
    sequence = [1, 2, 3, 4, 5, 6]
    task_prompt = "max-probability"
    role_prompt = "gpt-og"
    model_name = "gpt-3.5-turbo-0301"
    base = 2
    shots = 2
    shot_method = "random"
    prompt = create_explanation_prompt(
        sequence=sequence,
        task_prompt=task_prompt,
        role_prompt=role_prompt,
        model_name=model_name,
        base=base,
        shots=shots,
        shot_method=shot_method,
        seed=21,
    )
    print(prompt)


def test_self_consistency_evaluation():
    model_name = "text-davinci-003"
    sequence = [1, 2, 3]
    task_prompt = "self-consistency"
    base = 2
    shots = 4
    shot_method = "random"
    temperature = 0
    samples = 4

    outputs = self_consistency_evaluation(
        model_name=model_name,
        sequence=sequence,
        task_prompt=task_prompt,
        base=base,
        shots=shots,
        shot_method=shot_method,
        temperature=temperature,
        samples=samples,
        seed=0,
    )

    print(outputs)


if __name__ == "__main__":
    test_create_continuation_prompt()
    test_create_explanation_prompt()
