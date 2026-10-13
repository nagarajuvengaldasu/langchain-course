from typing import TypedDict, Annotated

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generate_chain, reflect_chain


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph):
    # The critic sees the whole thread as user input (Claude requires the conversation to end on a user turn)
    messages = [HumanMessage(content=m.text) for m in state["messages"]]
    res = reflect_chain.invoke({"messages": messages})
    return {"messages": [HumanMessage(content=res.text)]}


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue, path_map=[REFLECT, END])
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )
        ]
    }
    # Stream node updates so progress is visible (the full loop makes 7 LLM calls)
    final_tweet = None
    for step in graph.stream(inputs, stream_mode="updates"):
        for node, update in step.items():
            print(f"--- {node} done ---", flush=True)
            if node == GENERATE:
                final_tweet = update["messages"][-1].text

    print("\n===== FINAL TWEET =====")
    print(final_tweet)
