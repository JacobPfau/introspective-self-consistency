import datetime
import json

import openai

# This is ChatGPT!
MODEL = "gpt-3.5-turbo"

# This is the API key from your OpenAI account
openai.api_key_path = "../NYU_key.txt"

"""
    {"role": "system", "name": "example_user", "content": "200, 220, 240, "},
    {"role": "system", "name": "example_assistant", "content": "260"},
    {"role": "system", "name": "example_user", "content": "9, 81, 729, "},
    {"role": "system", "name": "example_assistant", "content": "6561"},

"""
messages_old = [
    {
        "role": "system",
        "content": """You are a mathematical assistant that predicts the next number in either an
                    arithmetic or geometric sequence accurately and concisely. You only respond with numbers.""",
    },
    {"role": "user", "content": "200, 220, 240, "},
    {"role": "assistant", "content": "260"},
    {"role": "user", "content": " 9, 81, 729, "},
    {"role": "assistant", "content": "6561"},
    {"role": "user", "content": "2, 4, "},
]

# Note: always use messages for the experiment, or saving will mess up
messages = [
    {
        "role": "system",
        "content": """You are a mathematical assistant that predicts the next number in either an
                    arithmetic or geometric sequence accurately and concisely. You only respond with numbers.""",
    },
    {"role": "user", "content": "200, 220, 240, "},
    {"role": "assistant", "content": "260"},
    {"role": "user", "content": "9, 81, 729, "},
    {"role": "assistant", "content": "6561"},
    {"role": "user", "content": "2, 4, "},
]
temperature = 1

results = dict()
for i in range(20):
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
    "model": MODEL,
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
