from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from openai import OpenAI, RateLimitError
import re

client = OpenAI(api_key=OPENAI_API_KEY)


# ----------------------------
# Tools
# ----------------------------

def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def search(query: str) -> str:
    knowledge_base = {
        "capital of france": "Paris",
        "largest planet": "Jupiter",
        "speed of light": "299,792,458 m/s"
    }
    return knowledge_base.get(query.lower(), "No results found")

TOOLS = {
    "calculator": calculator,
    "search": search
}

# ----------------------------
# Local fallback LLM
# ----------------------------

def local_fallback(messages):
    last_user_msg = messages[-1]["content"].lower()

    # Try to detect math expression (e.g., 2*3, 10+5)
    math_match = re.search(r"[\d\.\+\-\*\/\(\)]+", last_user_msg)

    if math_match:
        expr = math_match.group(0)
        try:
            return f"Final Answer: {eval(expr)}"
        except:
            pass

# 🔍 Knowledge fallback using search tool
    result = search(last_user_msg)
    if result != "No results found":
        return f"Final Answer: {result}"

    return "Final Answer: Sorry, I cannot process this right now (offline mode)."


# ----------------------------
# Prompt Template
# ----------------------------

SYSTEM_PROMPT = """
You are a ReAct agent.

You must follow this format exactly:

Thought: think about what to do
Action: one of [calculator, search]
Action Input: input for the tool

OR

Final Answer: the final answer to the user

Rules:
- Use tools when needed
- You can call multiple steps
- Be concise
"""

# ----------------------------
# Agent
# ----------------------------

class GPTReActAgent:
    def __init__(self):
        self.max_steps = 6

    def call_llm(self, messages):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0
            )
            return response.choices[0].message.content

        except RateLimitError:
            print("⚠️ API quota exceeded. Switching to local fallback...")
            return local_fallback(messages)

        except Exception as e:
            print(f"⚠️ API error: {e}. Using fallback...")
            return local_fallback(messages)

    def parse_action(self, text):
        action_match = re.search(r"Action:\s*(\w+)", text)
        input_match = re.search(r"Action Input:\s*(.*)", text)

        if not action_match:
            return None, None

        action = action_match.group(1)
        action_input = input_match.group(1) if input_match else ""

        return action, action_input

    def run(self, query):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]

        for step in range(self.max_steps):
            print(f"\nStep {step+1}")

            response = self.call_llm(messages)
            print(response)

            if "Final Answer:" in response:
                return response.split("Final Answer:")[-1].strip()

            action, action_input = self.parse_action(response)

            if action not in TOOLS:
                return f"Error: Unknown tool '{action}'"

            observation = TOOLS[action](action_input)
            print("Observation:", observation)

            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })

        return "Final Answer: Max steps reached."


# ----------------------------
# Run
# ----------------------------

if __name__ == "__main__":
    agent = GPTReActAgent()

    while True:
        query = input("\nAsk something (type 'stop' to exit): ")

        if query.lower() in ["stop", "exit", "quit"]:
            print("Exiting... 👋")
            break

        answer = agent.run(query)

        print("\n=== FINAL ANSWER ===")
        print(answer)
   