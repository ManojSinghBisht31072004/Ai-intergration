import os
import json
import math
from datetime import datetime
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

@tool
def calculator(expression: str) -> str:
    """Evaluates a math expression. Examples: '2 ** 15', 'math.sqrt(144)', '(45 * 3) / 2'"""
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def unit_converter(conversion: str) -> str:
    """Converts units. Input format: '<value> <from_unit> to <to_unit>'. Examples: '100 km to miles', '5 kg to pounds', '72 fahrenheit to celsius'"""
    try:
        parts = conversion.lower().split()
        value = float(parts[0])
        from_unit = parts[1]
        to_unit = parts[3]

        conversions = {
            ("km", "miles"): lambda v: v * 0.621371,
            ("miles", "km"): lambda v: v * 1.60934,
            ("kg", "pounds"): lambda v: v * 2.20462,
            ("pounds", "kg"): lambda v: v / 2.20462,
            ("celsius", "fahrenheit"): lambda v: (v * 9/5) + 32,
            ("fahrenheit", "celsius"): lambda v: (v - 32) * 5/9,
            ("meters", "feet"): lambda v: v * 3.28084,
            ("feet", "meters"): lambda v: v / 3.28084,
            ("liters", "gallons"): lambda v: v * 0.264172,
            ("gallons", "liters"): lambda v: v / 0.264172,
        }

        key = (from_unit, to_unit)
        if key in conversions:
            result = conversions[key](value)
            return f"{value} {from_unit} = {round(result, 4)} {to_unit}"
        else:
            return f"Conversion from {from_unit} to {to_unit} not supported."
    except Exception as e:
        return f"Error: {e}"


def build_agent():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )

    tools = [calculator, unit_converter]

    # Inject current date directly into system prompt
    current_date = datetime.now().strftime("%A, %B %d, %Y. Current time: %H:%M:%S")

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a helpful AI assistant with access to tools.

Today's date and time is: {current_date}

Rules:
- Math or calculations → use calculator tool
- Unit conversion → use unit_converter tool
- Questions about today's date or current time → answer directly using the date above
- General knowledge → answer directly without any tool

Be concise and accurate."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
    return executor


def run_agent(agent, user_message: str) -> dict:
    print(f"\n{'='*60}")
    print(f"PROMPT: {user_message}")
    print('='*60)

    result = agent.invoke({"input": user_message})

    tools_used = []
    for step in result.get("intermediate_steps", []):
        action = step[0]
        if hasattr(action, "tool"):
            tools_used.append(action.tool)

    structured = {
        "prompt": user_message,
        "answer": result["output"],
        "tools_used": tools_used if tools_used else ["none — answered directly"]
    }

    print(f"\nSTRUCTURED RESPONSE:")
    print(json.dumps(structured, indent=2))
    return structured


if __name__ == "__main__":
    agent = build_agent()

    test_prompts = [
        "What is 2 to the power of 15?",
        "What is today's date?",
        "Convert 100 km to miles",
        "If I have 8.5 kg of flour, how many pounds is that?",
        "What is the square root of 1764?",
        "What is the capital of France?",
        "Explain what photosynthesis is in simple terms",
        "What are the three laws of motion?",
        "Who wrote the novel 1984?",
        "What does HTTP stand for?",
    ]

    results = []
    for prompt in test_prompts:
        result = run_agent(agent, prompt)
        results.append(result)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n\nAll results saved to results.json")
    