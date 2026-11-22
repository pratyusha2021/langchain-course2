from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# --- Tools (LangChain @tool decorator) ---


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# --- Agent Loop ---


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")






# from dotenv import load_dotenv
# load_dotenv()   

# from langchain.chat_models import init_chat_model
# from langchain.tools import tool
# from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
# from langsmith import traceable

# MAX_ITERATIONS = 3
# MODEL = "qwen3:1.7b"

# @tool
# def get_product_price(product: str) -> float:
#     """
#         Look up the price of a product in the catalog.
#     """
#     print(f"Looking up price for product:(product='{product}')")
#     prices = {
#         "laptop": 999.99,
#         "smartphone": 699.99,
#         "headphones": 199.99,
#         "tablet": 499.99,
#         "smartwatch": 299.99,
#     }
#     return prices.get(product, 0)

# @tool
# def get_discount(price: float, discount_tier: str) -> float:        
#     """
#     Apply a discount tier to a price and return the final price. Available discount tiers are: gold, silver, bronze.
#     """
#     print(f"Calculating discounted price for price:(price='{price}', discount_tier='{discount_tier}')")
#     discount_percentages = {
#         "gold": 0.20,
#         "silver": 0.10,
#         "bronze": 0.05,
#     }
#     discount = discount_percentages.get(discount_tier, 0)
#     final_price = price * (1 - discount)
#     return final_price

# # Agent loop

# @traceable(name = "LangChain Agent Loop")
# def run_agent_loop(user_question: str):
#     tools = [get_product_price, get_discount]
#     tools_dict = {t.name: t for t in tools}

#     llm = init_chat_model(f"ollama:{MODEL}", temperature=0.0)
#     llm_with_tools = llm.bind_tools(tools)

#     print(f"User question: {user_question}")
#     print("=" * 60)

#     messages = [
#         SystemMessage(
#             content=(
#                 "You are a helpful assistant that can answer questions about product prices and discounts. "
#                 "Use the tools provided to look up product prices and apply discounts. "
#                 "You have access to the following tools: "
#                 "1. get_product_price(product: str) -> float: Look up the price of a product in the catalog. "
#                 "2. get_discount(price: float, discount_tier: str) -> float: Apply a discount tier to a price and return the final price. "
#                 "Available discount tiers are: gold, silver, bronze. "
#                 "Use these tools to answer the user's question."
#                 "Strict rules for the agent: "
#                 "never make up prices or discounts."
#             )
#         ),
#         HumanMessage(content=user_question),
#     ]

#     for iteration in range(1, MAX_ITERATIONS + 1):
#         print(f"Iteration {iteration}:")
#         ai_message = llm_with_tools.invoke(messages)
#         tool_call = ai_message.tool_calls
#         print(f"AI: {ai_message.content}")
#         # if no tool call required
#         if not tool_call:
#             print("Final answer:")
#             print(ai_message.content)
#             return ai_message.content

#         # process only first tool call - force single tool call per iteration
#         tool_call = tool_call[0]
#         tool_name = tool_call.get("name")
#         tool_args = tool_call.get("args", {})
#         tool_id = tool_call.get("id")

#         print(f"Tool Selected: {tool_name} with args: {tool_args}")
#         tool_to_use = tools_dict.get(tool_name)
#         if not tool_to_use:
#             raise ValueError(f"Tool '{tool_name}' not found in available tools.")

#         observation = tool_to_use.invoke(tool_args)
#         print(f"Observation: {observation}")

# if __name__ == "__main__":
#     print("hello agent (.bind_tools)!")
#     result = run_agent_loop("I want to buy a laptop and I have a gold discount tier. What is the final price?")


    



