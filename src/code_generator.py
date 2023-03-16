from human_eval.data import read_problems, write_jsonl

from src.models.openai import generate_completion  # type: ignore

problems = read_problems()

print(len(problems))

num_samples_per_task = 1

samples = [
    dict(
        task_id=task_id,
        completion=generate_completion(problems[task_id]["prompt"]),
    )
    for i, task_id in enumerate(problems)
    for _ in range(num_samples_per_task)
    if i < 10
]
write_jsonl("samples.jsonl", samples)
