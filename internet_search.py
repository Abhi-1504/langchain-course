import os

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
#from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schema import AgentResponse


load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")  # For OpenAI GPT Model
structured_llm = llm.with_structured_output(AgentResponse)
#llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")  # for Google Gemini Model
react_prompt = hub.pull("hwchase17/react")
#output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"]
).partial(format_instructions="")#output_parser.get_format_instructions())

agent = create_react_agent(llm, tools=tools, prompt=react_prompt_with_format_instructions)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x['output'])
#parse_output = RunnableLambda(lambda x: output_parser.parse(x))
chain = agent_executor | extract_output | structured_llm #parse_output

def main():
    result = chain.invoke(
        input={
            "input": "search for 3 job postings for an ai engineer using langchain in the Bengaluru on linkedin and list their details"
        }
    )

    print(result)


if __name__ == "__main__":
    main()
