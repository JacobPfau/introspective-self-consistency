from src.models.openai_model import OpenAITextModels, generate_logprob_response_with_turns
from src.evals.q12_logprob_dependency import run_q1_2_eval

model = OpenAITextModels.TEXT_DAVINCI_003
_, log_probs_dog = generate_logprob_response_with_turns(
    model, [
        {"content": "Q: What is cuter A: Cats or B: Dogs? A: Dogs"},
    ],
    max_tokens=0
)
_, log_probs_cat = generate_logprob_response_with_turns(
    model, [
        {"content": "Q: What is cuter A: Cats or B: Dogs? A: Cats"},
    ],
    max_tokens=0
)

_, log_probs_magikarp = generate_logprob_response_with_turns(
    model, [
        {"content": "Q: What is cuter A: Cats or B: Dogs? A: SolidGoldMagikarp"},
    ],
    max_tokens=0
)
print(log_probs_dog["token_logprobs"][log_probs_dog['tokens'].index(' Dogs')])  # -0.10365415
print(log_probs_cat["token_logprobs"][log_probs_cat['tokens'].index(' Cats')])  # -9.249667
print(log_probs_magikarp["token_logprobs"][log_probs_magikarp['tokens'].index(' SolidGoldMagikarp')])  # -29.390312
