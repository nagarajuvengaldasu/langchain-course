from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer:str = Field(description="there agents answer to the query")
    sources: List[Source] = Field(default_factory=list , description="List of Sources used to generate the answer")
llm = ChatAnthropic(model="claude-sonnet-5-5")
tools = [TavilySearch()]
agent = create_agent(model = llm , tools = tools , response_format=AgentResponse)

def main():
    print("Hello from langchain course")
    result = agent.invoke({"messages": [HumanMessage(content="search for 3 jobs the role is ai enginner from the linkedin")]})
    print(result["messages"][-1].content)
if __name__ == "__main__":
    main()