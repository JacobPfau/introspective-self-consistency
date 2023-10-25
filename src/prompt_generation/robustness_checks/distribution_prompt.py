"""
Create a dictionary of distributions we might use for prompting explanations / continuations.

Read all _task_ prompts from
 - ./<task_prompt>/continuation.txt
 - ./<task_prompt>/explanation.txt

 and store them in a dictionary of the form:
 TASK_PROMPTS = {
        <task_prompt>: {
            "continuation": Path(./<task_prompt>/continuation.txt).read_text(),
            "explanation": Path(./<task_prompt>/explanation.txt).read_text(),
        }
    }
"""
from collections import defaultdict
from pathlib import Path

TASK_PROMPTS = defaultdict(dict)
HERE = Path(__file__).parent

for file in HERE.glob("**/continuation.txt"):
    task_prompt = file.parent.name
    TASK_PROMPTS[task_prompt]["continuation"] = file.read_text()


for file in HERE.glob("**/explanation.txt"):
    task_prompt = file.parent.name
    TASK_PROMPTS[task_prompt]["explanation"] = file.read_text()

TASK_PROMPTS = dict(TASK_PROMPTS)

# todo: same for _role_ prompts

ROLE_PROMPTS = defaultdict(dict)

for file in HERE.glob("**/role.txt"):
    role_prompt = file.parent.name
    ROLE_PROMPTS[role_prompt] = file.read_text()

ROLE_PROMPTS = dict(ROLE_PROMPTS)
