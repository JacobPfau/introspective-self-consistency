import numpy as np
from tqdm.auto import tqdm
import sys

sys.path.insert(0, "..")

from src.models.openai_model import generate_chat_completion
from src.pipelines.sequence_completions import find_ambigious_integer_sequences, generate_sequence_completion_prompt

MAX_OFFSET = 8

ambigious_sequences = find_ambigious_integer_sequences()
accs = []
for sequence, fns in tqdm(ambigious_sequences.items()):
    fn = fns[0] # just select one for now
    completion_prompt = generate_sequence_completion_prompt(sequence, fn, n_shots=8, use_cot=True)
    explanation_prompt = generate_sequence_completion_prompt(sequence, fn, n_shots=8, use_cot=True, prompt_type="explanation")

    completion_resp = generate_chat_completion(completion_prompt['prompt_turns'])
    explanation_resp = generate_chat_completion(explanation_prompt['prompt_turns'])
    
    explanation = explanation_resp.split('\n')[0].strip()
    actual_completion = completion_resp.split('\n')[0].strip()

    try:
        eval(explanation)(0)
    except:
        continue

    # find the offset that generates the sequence
    sequence = [int(item) for item in sequence.split(',')]
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
        accs.append(0)
        continue

    last_completion = eval(explanation)(last_completion_step + 1)

    accs.append(
        int(actual_completion) == last_completion
    )

print(
    f"""
    Evaluated {len(ambigious_sequences.items())} ambigious sequences.
    Resulting in {round(np.mean(accs), 2)}% accuracy.
    """
)