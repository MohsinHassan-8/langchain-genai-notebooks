from langchain_community.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

@tool
def multiply(a : int, b : int) -> int:
       '''Given 2 numbers, a and b, this tool will multiply those numbers and return the result'''
       return a * b

result = multiply.invoke({'a' : 3, 'b' : 5})

print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.args_schema.model_json_schema)

