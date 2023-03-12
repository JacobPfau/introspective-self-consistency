import openai

# This is ChatGPT!
MODEL = "gpt-3.5-turbo"

# This is the API key from your OpenAI account
openai.api_key_path = "../NYU_key.txt"

response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "Who's there?"},
        {"role": "user", "content": "Orange."},
    ],
    temperature=1,
)

print(response["choices"][0]["message"]["content"])
