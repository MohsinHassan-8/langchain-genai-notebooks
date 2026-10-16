from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

@tool
def multiply(a : int, b : int) -> int:
       '''Given 2 numbers a and b, this tool returns their product'''
       return a*b

print(multiply.invoke({'a' : 2, 'b' : 4}))

model_with_tools = model.bind_tools([multiply])
