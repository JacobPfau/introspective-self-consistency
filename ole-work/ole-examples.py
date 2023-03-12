import datetime
import json

import openai

# This is ChatGPT!
MODEL = "gpt-3.5-turbo"

# This is the API key from your OpenAI account
openai.api_key_path = "../NYU_key.txt"

"""
The task is to output 'X' if the sentence contains a [category withheld] and 'Y' otherwise.

Sentence: "The worm is in the meadow" -> output: 'X'
Sentence: "The student is in the museum" -> output: 'Y'
Sentence: "The duck is in the canyon" -> output:

"""
messages_old = [
    {
        "role": "user",
        "content": """You are a larconic assistant that predicts the next number in either an
                      arithmetic or geometric sequence accurately and concisely.""",
    },
    {
        "role": "assistant",
        "content": """I am a larconic assistant that predicts the next number in either an
                      arithmetic or geometric sequence accurately and concisely.""",
    },
    {"role": "user", "content": "1, 3, 9, "},
    {"role": "assistant", "content": "27"},
    {"role": "user", "content": " 4, 8, 12, "},
    {"role": "assistant", "content": "16"},
    {"role": "user", "content": "2, 4, "},
]

messages = [
    {
        "role": "system",
        "content": """You are a larconic assistant that predicts the next number in either an
                    arithmetic or geometric sequence accurately and concisely.""",
    },
    {"role": "system", "name": "example_user", "content": "1, 3, 9, "},
    {"role": "system", "name": "example_assistant", "content": "27"},
    {"role": "system", "name": "example_user", "content": " 4, 8, 12, "},
    {"role": "system", "name": "example_assistant", "content": "16"},
    {"role": "system", "name": "example_user", "content": "2, 4, "},
]
temperature = 1

results = dict()
for i in range(50):
    # Collect the results
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
    )
    result = response["choices"][0]["message"]["content"]
    if result in results:
        results[result] += 1
    else:
        results[result] = 1

total_info = {
    "temperature": temperature,
    "messages": messages,
    "results": results,
}

# Save the results, prompt, and temperature as a json file
# Use datetime to create a unique file name

now = datetime.datetime.now()
# Format the date
now = now.strftime("%Y-%m-%d_%H-%M-%S")
file_name = "ole-data/" + f"ole_experiment_{now}.json"
with open(file_name, "w+") as f:
    json.dump(total_info, f)

print(results)
