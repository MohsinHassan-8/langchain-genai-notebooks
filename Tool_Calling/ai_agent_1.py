from langchain.agents import create_agent

from langchain_community.tools import DuckDuckGoSearchRun

from langchain_google_genai  import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()



model = ChatGoogleGenerativeAI( model = 'gemini-2.5-flash')
search_tool = DuckDuckGoSearchRun()

agent = create_agent(
       model = model,
       tools = [search_tool],
       system_prompt="You are a helpful AI assistant."
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "3 ways to reach Lahore from Karachi"
            }
        ]
    }
)

print(response['messages'][-1].content)

