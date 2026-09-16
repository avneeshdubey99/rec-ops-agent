"""One place to create the LLM, so every node uses the same model settings.

Default: Anthropic Claude. Set LLM_PROVIDER=openai in .env to use OpenAI instead.
"""
import os

from dotenv import load_dotenv

load_dotenv()

PROVIDERS = {
    # provider: (API key env var, default model)
    "anthropic": ("ANTHROPIC_API_KEY", "claude-sonnet-5"),
    "openai": ("OPENAI_API_KEY", "gpt-4o-mini"),
}


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "anthropic").strip().lower()


def missing_key() -> str | None:
    """Name of the required API key env var if it isn't set, else None."""
    key_var = PROVIDERS.get(provider(), PROVIDERS["anthropic"])[0]
    return None if os.getenv(key_var) else key_var


def get_llm():
    name = provider()
    if name == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("LLM_MODEL", PROVIDERS["openai"][1]), temperature=0)

    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model=os.getenv("LLM_MODEL", PROVIDERS["anthropic"][1]),
        temperature=0,
        max_tokens=4096,
        max_retries=2,
    )
