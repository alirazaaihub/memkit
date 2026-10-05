"""memkit — drop-in short-term + long-term memory for chatbots and agents.

    from memkit import Memory

    mem = Memory(
        storage_dir="./agent_memory",
        llm=my_llm,                    # callable: llm(messages) -> str
        llm_tools=my_llm_with_tools,   # callable: llm_tools(messages, tools) -> assistant msg
        max_tokens_stm=4000,
        max_tokens_ltm=8000,
    )

    mem.add("user", "hello")
    messages = [{"role": "system", "content": mem.system_prompt_fragment()}] + mem.context()
    # ... call your own LLM with `messages`, then record its reply:
    mem.add("assistant", reply_text)
    mem.close()
"""

from memkit.facade import Memory
from memkit.llm import AnthropicLLM, OpenAILLM, adapt_llm, messages_to_openai

__version__ = "0.1.0"

__all__ = ["Memory", "OpenAILLM", "AnthropicLLM", "adapt_llm",
           "messages_to_openai", "__version__"]
