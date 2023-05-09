from models.openai_model import CHAT_MODEL_NAME, DAVINCI_MODEL_NAME
from q11.prompts.continuation_prompt import create_continuation_prompt
from q11.prompts.explanation_prompt import create_explanation_prompt
from q11.evals.check_self_consistency import self_consistency_evaluation


def test_completion_prompt():
    sequence = [1, 2, 3, 4, 5]
    distribution = "default"
    model_name = "CHAT"
    base = 2
    prompt = create_continuation_prompt(
        sequence=sequence,
        distribution=distribution,
        model_name=model_name,
        base=base,
    )
    print(prompt)
    print("buggo")


def test_create_continuation_prompt():
    sequence = [1, 2, 3, 4, 5, 6]
    distribution = "default"
    model_name = "CHAT"
    base = 10
    shots = 5
    shot_method = "random"
    prompt = create_continuation_prompt(
        sequence=sequence,
        distribution=distribution,
        model_name=model_name,
        base=base,
        shots=shots,
        shot_method=shot_method,
    )
    print(prompt)


def test_create_explanation_prompt():
    sequence = [1, 2, 3, 4, 5, 6]
    distribution = "default"
    model_name = "CHAT"
    base = 10
    shots = 5
    shot_method = "random"
    prompt = create_explanation_prompt(
        sequence=sequence,
        distribution=distribution,
        model_name=model_name,
        base=base,
        shots=shots,
        shot_method=shot_method,
    )
    print(prompt)


def test_self_consistency_evaluation():
    model_name = "DAVINCI"
    sequence = [1, 2, 3]
    distribution = "default"
    base = 10
    shots = 4
    shot_method = "random"
    temperature = 0
    samples = 4

    outputs = self_consistency_evaluation(
        model_name=model_name,
        sequence=sequence,
        distribution=distribution,
        base=base,
        shots=shots,
        shot_method=shot_method,
        temperature=temperature,
        samples=samples,
    )

    print(outputs)


def test_create_continuation_prompt():
    sequence = [1, 2, 3, 4, 5, 6]
    distribution = "default"
    model_name = "DAVINCI"
    base = 2
    shots = 5
    shot_method = "random"
    prompt = create_continuation_prompt(
        sequence=sequence,
        distribution=distribution,
        model_name=model_name,
        base=base,
        shots=shots,
        shot_method=shot_method,
    )
    print(prompt)


def test_create_explanation_prompt():
    sequence = [1, 2, 3, 4, 5, 6]
    distribution = "default"
    model_name = "CHAT"
    base = 2
    shots = 5
    shot_method = "random"
    prompt = create_explanation_prompt(
        sequence=sequence,
        distribution=distribution,
        model_name=model_name,
        base=base,
        shots=shots,
        shot_method=shot_method,
    )
    print(prompt)


def test_self_consistency_evaluation():
    model_name = "DAVINCI"
    sequence = [1, 2, 3]
    distribution = "default"
    base = 2
    shots = 4
    shot_method = "random"
    temperature = 0
    samples = 4

    outputs = self_consistency_evaluation(
        model_name=model_name,
        sequence=sequence,
        distribution=distribution,
        base=base,
        shots=shots,
        shot_method=shot_method,
        temperature=temperature,
        samples=samples,
    )

    print(outputs)


if __name__ == "__main__":
    test_self_consistency_evaluation()
