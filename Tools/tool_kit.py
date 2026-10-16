from langchain_community.tools import tool

@tool
def mult( a : int, b : int) -> int :
       'multiply these 2 numbers'
       return a * b

@tool
def add(a : int, b : int) -> int :
       'add these 2 numbers'
       return a + b

class Tool_kit():
       def get_tools(self):
              return [add,mult]

toolkit = Tool_kit()
tools_1 = toolkit.get_tools()

for tools in tools_1:
       print(tools.name, '->', tools.description)

