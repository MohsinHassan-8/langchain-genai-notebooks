from langchain_community.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

@tool
def multiply(a:int, b:int) -> int:
       '''Multiply 2 numbers'''
       return a * b

model_with_tools = model.bind_tools([multiply])

result = model_with_tools.invoke('multiply 2 with 3')
#print(result.tool_calls)
print(result.tool_calls[0])
print(result.tool_calls[0]['args'])
print(multiply.invoke(result.tool_calls[0]['args']))
print(multiply.invoke(result.tool_calls[0]))
