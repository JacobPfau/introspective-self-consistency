"""
Generate a prompt which asks the model to choose from some distribution of functions.
"""

SELF_CONTINUATION_PROMPT = (
    "In your opinion, what is the most likely continuation of the following sequence?"
)

SELF_EXPLANATION_PROMPT = """In your opinion, what is the most likely explanation for how the following sequence was generated?
Give your response as a function."""
