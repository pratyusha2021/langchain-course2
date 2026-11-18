from dotenv import load_dotenv
from langchain_openai import tools

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
# from tavily import TavilyClient
from langchain_tavily import TavilySearch

# tavily = TavilyClient()

# @tool
# def SearchAgent(query):
#     """
#     Tool that searches the web for information and returns the results.
#     Args:
#         query : The search query.
#     Returns:
#         str: The search results.
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)



# llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm = ChatOllama(model = "llama3.2")
# tools = [SearchAgent]
tools = [TavilySearch()]
agent = create_agent(llm,tools)



def main() -> None:
    print("Hello from langchain-course2!")
    result = agent.invoke({"messages": [HumanMessage(content="Search for 3 job in linkedin for AI Engineer with 3 years of experience in india")]})
    print(result)


if __name__ == "__main__":
    main()
