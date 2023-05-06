import pytest
import tiktoken

from models.openai_model import (
    OpenAIChatModels,
    OpenAITextModels,
    generate_chat_completion,
    generate_text_completion,
)


@pytest.mark.parametrize("model", OpenAITextModels.list())
def test_generate_text_completion(model):
    test_prompt = "Once upon a time,"
    max_tokens = 5
    text = generate_text_completion(test_prompt, model=model, max_tokens=max_tokens)
    tokens = tiktoken.encoding_for_model(model).encode(text)
    assert len(tokens) == max_tokens
    assert (
        len(set(tokens)) == max_tokens
    )  # sanity check: assume all tokens are unique (reasonable for short text)


@pytest.mark.parametrize("model", OpenAIChatModels.list())
def test_generate_chat_completion(model):
    max_tokens = 5
    test_prompt = [
        {"role": "user", "content": "Hello, how are you?"},
    ]
    text = generate_chat_completion(test_prompt, model=model, max_tokens=max_tokens)
    tokens = tiktoken.encoding_for_model(model).encode(text)
    assert len(tokens) == max_tokens
    assert (
        len(set(tokens)) == max_tokens
    )  # sanity check: assume all tokens are unique (reasonable for short text)
