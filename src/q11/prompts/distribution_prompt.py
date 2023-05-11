"""
Generate a prompt which asks the model to choose from some distribution of functions.
"""

SELF_CONTINUATION_PROMPT = """In your opinion, what is the most likely continuation of the following sequence?
 Give your response as a number, with no additional commentary."""

SELF_EXPLANATION_PROMPT = """In your opinion, what is the most likely explanation for how the following sequence was generated?
Give your response in the format:
Explanation: <function>
Offset: <offset>

Where <function> refers to the function used to generate the sequence,
and <offset> refers to the first integer used to generate the sequence.
Furthermore, assume that the function is applied to a sequence of consecutive integers, starting at the offset.
For example, if the sequence is 1, 2, 3, 4, 5, then the function is lambda x: x, then the offset is 1.
The output of the function should be the number expressed as a decimal integer.
"""
