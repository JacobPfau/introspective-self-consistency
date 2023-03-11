import os

import openai
from human_eval.data import read_problems, write_jsonl  # type: ignore

MODEL_NAME = "text-davinci-003"
# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_one_code_completion(
    prompt: str, temperature: int = 0, max_tokens: int = 256
) -> str:
    # docs: https://platform.openai.com/docs/api-reference/completions/create
    response = openai.Completion.create(
        model=MODEL_NAME, prompt=prompt, temperature=temperature, max_tokens=max_tokens
    )

    if len(response["choices"]) > 0:
        return response["choices"][0]["text"]

    else:
        raise KeyError("Response did not return enough `choices`")


problems = read_problems()

print(len(problems))

num_samples_per_task = 1

samples = [
    dict(
        task_id=task_id,
        completion=generate_one_code_completion(problems[task_id]["prompt"]),
    )
    for i, task_id in enumerate(problems)
    for _ in range(num_samples_per_task)
    if i < 10
]
write_jsonl("samples.jsonl", samples)
