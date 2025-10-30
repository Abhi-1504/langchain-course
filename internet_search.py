import os
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_tavily import TavilySearch

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model='gpt-4') # For OpenAI GPT Model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro") # for Google Gemini Model
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm,
    tools=tools,
    prompt=react_prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor

def main():
    
    #llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    result = chain.invoke(
        input={
            "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
            }
        )
    
    print(result)

if __name__ == "__main__":
    main()