"""
Generate a prompt which asks the model to choose from some distribution of functions.
"""

SELF_CONTINUATION_PROMPT = "In your opinion, what is the most likely continuation of the following sequence? Give your response as a number, with no additional commentary."

SELF_EXPLANATION_PROMPT = """In your opinion, what is the most likely function which generated the following sequence?
Assume the first number was generated by f(0), the second by f(1), and so on.
The function should output a binary string prefixed by 0b, i.e. '0b1010'.
Give your response in the format:
Explanation: <function>
"""
