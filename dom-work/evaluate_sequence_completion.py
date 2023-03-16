import sys

import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, "..")

from src.models.openai_model import generate_chat_completion
from src.pipelines.sequence_completions import (
    find_ambigious_integer_sequences,
    generate_sequence_completion_prompt,
)

MAX_OFFSET = 8
NUM_SHOTS = 8
COT = True


def sequence_completion_equality(sequence, fn):
    completion_prompt = generate_sequence_completion_prompt(
        sequence, fn, n_shots=NUM_SHOTS, use_cot=COT
    )
    explanation_prompt = generate_sequence_completion_prompt(
        sequence, fn, n_shots=NUM_SHOTS, use_cot=COT, prompt_type="explanation"
    )

    completion_resp = generate_chat_completion(completion_prompt["prompt_turns"])
    explanation_resp = generate_chat_completion(explanation_prompt["prompt_turns"])

    explanation = explanation_resp.split("\n")[0].strip()
    actual_completion = completion_resp.split("\n")[0].strip()

    # find the offset that generates the sequence
    sequence = [int(item) for item in sequence.split(",")]
    last_completion_step = None
    sequence_matched = []
    for i in range(MAX_OFFSET):
        completion = eval(explanation)(i)
        if completion in sequence:
            sequence_matched.append(completion)
        if sequence_matched == sequence:
            last_completion_step = i
            break

    if last_completion_step is None:
        return 0

    last_completion = eval(explanation)(last_completion_step + 1)

    return int(actual_completion) == last_completion


ambigious_sequences = find_ambigious_integer_sequences()
accs = []
for sequence, fns in tqdm(ambigious_sequences.items()):
    fn = fns[0]  # just select one for now
    try:
        accs.append(sequence_completion_equality(sequence, fn))
    except Exception as e:
        print(e)
        accs.append(0)


print(
    f"""
    Evaluated {len(ambigious_sequences.items())} ambigious sequences.
    Resulting in {round(np.mean(accs), 2)}% accuracy.
    """
)
