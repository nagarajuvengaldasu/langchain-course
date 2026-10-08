import importlib

langchain_agent = importlib.import_module("1_agent_loop_langchain_tool_calling")
raw_agent = importlib.import_module("2_agent_loop_raw_function_calling")
react_agent = importlib.import_module("3_raw_react_prompt")

QUESTION = "What is the price of a laptop after applying a gold discount?"

def main():
    print("Hello from 02-agents-under-hood!")

    # print("\n######## 1. LangChain tool calling (.bind_tools) ########")
    # result_1 = langchain_agent.run_agent(QUESTION)

    # print("\n######## 2. Raw function calling (ollama.chat) ########")
    # result_2 = raw_agent.run_agent(QUESTION)

    print("\n######## 3. Raw ReAct prompt (regex parsing) ########")
    result_3 = react_agent.run_agent(QUESTION)

    print("\n######## Results ########")
    # print(f"LangChain : {result_1}")
    # print(f"Raw Ollama: {result_2}")
    print(f"ReAct     : {result_3}")

if __name__ == "__main__":
    main()