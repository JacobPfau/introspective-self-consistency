SYSTEM_PROMPT = """
You are a mathematical assistant.
You are helping with integer sequences like arithmetic or geometric sequences.
The sequence's are not always 0 indexed, some are offset starting from an arbitrary i-index value.
You answer accurately and concisely.
"""

SYSTEM_PROMPT_COMPLETION = """
You only respond with numbers.
"""

SYSTEM_PROMPT_EXPLANATION = """
You only respond with code.
"""

BASE_PROMPT = """
For the sequence: {}
"""

BASE_PROMPT_COMPLETION = """
Complete the next number and only the next number.
"""

BASE_PROMPT_EXPLANATION = """
Give the code that generates the above sequence.
"""

BASE_PROMPT_EXPLANATION_MULTIPLE_CHOICE = """
Select the code that generates the above sequence from the following options.
Only respond with the number of the valid option.
Options:
"""

COT_PROMPT = """
Let's solve this step by step:
"""

COT_STEP = """
Step {}:
>>> fn = {}; fn({})
{}
"""
