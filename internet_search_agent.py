import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from schema import AgentResponse


load_dotenv()

tools = [TavilySearch()]
model = ChatOpenAI(model='gpt-4')

agent = create_agent(model,
                     tools=tools,
                     response_format=AgentResponse
                     )

def main():
    result = agent.invoke(
        {
            "messages": [
                {   
                    "role" : "user",
                    "content": "search for 3 job postings for an ai engineer using langchain in the Bengaluru on linkedin and list their details"
                }
            ]
        }
    )

    print(result['structured_response'])


if __name__ == "__main__":
    main()
