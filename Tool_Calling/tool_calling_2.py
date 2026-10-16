from langchain_community.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

@tool
def multiply(a:int, b:int) -> int:
       '''Multiply 2 numbers'''
       return a * b

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

model_with_tool =model.bind_tools([multiply])

messages = []

query = HumanMessage('multiply 2 with 3')
messages.append(query)

result = model_with_tool.invoke(messages)
messages.append(result)

tool_result = multiply.invoke(result.tool_calls[0])
messages.append(tool_result)

print(type(tool_result))
print(model_with_tool.invoke(messages).content)

