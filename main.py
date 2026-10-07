from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch


llm = ChatAnthropic(model="claude-sonnet-5-5")
tools = [TavilySearch()]
agent = create_agent(model = llm , tools = tools)

def main():
    print("Hello from langchain course")
    result = agent.invoke({"messages": [HumanMessage(content="search for 3 jobs the role is ai enginner from the linkedin")]})
    print(result["messages"][-1].content)
if __name__ == "__main__":
    main()