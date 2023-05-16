"""
Create a dictionary of distributions we might use for prompting explanations / continuations.
"""
from src.evals.prompts.distribution_prompt import (
    SELF_CONTINUATION_PROMPT,
    SELF_EXPLANATION_PROMPT,
)

DISTRIBUTIONS = {
    "default": {
        "continuation": SELF_CONTINUATION_PROMPT,
        "explanation": SELF_EXPLANATION_PROMPT,
    }
}
