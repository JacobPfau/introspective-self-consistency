from typing import List, Union

from src.models import anthropic_model, openai_model


def generate_response_with_turns(
    model: Union[
        str,
        anthropic_model.AnthropicChatModels,
        openai_model.OpenAIChatModels,
        openai_model.OpenAITextModels,
    ],
    turns: List[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> str:
    if (
        model in openai_model.OpenAITextModels.list()
        or model in openai_model.OpenAIChatModels.list()
    ):
        return openai_model.generate_response_with_turns(
            turns=turns,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
    elif model in anthropic_model.AnthropicChatModels.list():
        return anthropic_model.generate_chat_completion(
            prompt_turns=turns,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
    else:
        raise ValueError(f"Invalid model: {model}")
