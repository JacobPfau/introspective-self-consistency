from src.prompt_generation import PromptBase, get_formatted_prompt

MODEL_CONSISTENCY_SHOTS = [
    {"sequence": "1, 2, 3", "fn": "lambda x: x + 1", "answer": "Y"},
    {"sequence": "3, 3, 4", "fn": "lambda x: 3 * x + 2", "answer": "N"},
    {"sequence": "8, 18, 32", "fn": "lambda x: 2 * x ** 2", "answer": "Y"},
    {"sequence": "256, 1024", "fn": "lambda x: 4 ** x", "answer": "Y"},
    {"sequence": "7, 5, 12", "fn": "lambda x: (3 * x) | 4", "answer": "N"},
    {"sequence": "2, 3, 0, 1", "fn": "lambda x: (x * 5) % 4", "answer": "Y"},
]


MODEL_COMPLETION_SHOTS = [
    {"sequence": "1, 2, 3", "fn": "lambda x: x + 1", "answer": 4},
    {"sequence": "2, 5, 8", "fn": "lambda x: 3 * x + 2", "answer": 11},
    {"sequence": "8, 18, 32", "fn": "lambda x: 2 * x ** 2", "answer": 50},
    {"sequence": "256, 1024", "fn": "lambda x: 4 ** x", "answer": 4096},
    {"sequence": "7, 6, 13", "fn": "lambda x: (3 * x) | 4", "answer": 12},
    {"sequence": "2, 3, 0, 1", "fn": "lambda x: (x * 5) % 4", "answer": 2},
]


def generate_consistency_check_prompt(sequence: str, fn: str) -> str:
    prompt = ""
    for shot in MODEL_CONSISTENCY_SHOTS:
        prompt += get_formatted_prompt(
            PromptBase.BASE_CONSISTENCY, {"seq": shot["sequence"], "fn": shot["fn"]}
        )
        prompt += " " + shot["answer"]
    prompt += get_formatted_prompt(
        PromptBase.BASE_CONSISTENCY, {"seq": sequence, "fn": fn}
    )
    return prompt


def generate_completion_check_prompt(sequence, fn) -> str:
    prompt = ""
    for shot in MODEL_COMPLETION_SHOTS:
        prompt += get_formatted_prompt(
            PromptBase.CONSISTENCY_COMPLETION,
            {"seq": shot["sequence"], "fn": shot["fn"]},
        )
        prompt += " " + str(shot["answer"])
    prompt += get_formatted_prompt(
        PromptBase.CONSISTENCY_COMPLETION, {"seq": sequence, "fn": fn}
    )
    return prompt
