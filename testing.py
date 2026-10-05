import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("TOKENHARBOR_API_KEY"),
    base_url=os.getenv("BASE_URL", "https://tokenharbor.ai/v1"),
)
MODEL = os.getenv("MODEL_NAME", "deepseek-v4.1-flash:free")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print(f"Chatbot ready (model: {MODEL}). Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ("exit", "quit"):
        print("Bye!")
        break

    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(model=MODEL, messages=messages)
        reply = response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}\n")
        messages.pop()  # failed message history se hata do
        continue

    messages.append({"role": "assistant", "content": reply})
    print(f"Bot: {reply}\n")