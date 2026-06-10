import os

from .interface import LLM

from dotenv import load_dotenv

load_dotenv()

if os.environ.get("OPENAI_API_KEY"):
    from .openai_adapter import OpenAILLM as DefaultLLM

    default_llm = DefaultLLM()
else:
    raise Exception("Error: No LLM API keys provided.")
