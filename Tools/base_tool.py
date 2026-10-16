from langchain_community.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class MultiplyStruc(BaseModel):
       a : int = Field( description='First number to add' )
       b : int = Field( description='Second number to add' )

class MultiplyTool(BaseTool):
       name : str = 'Multiply'
       description : str = 'multiply 2 numbers'
       args_schema: Type[BaseModel] = MultiplyStruc

       def _run(self, a : int, b : int) -> int:
              return a * b
       
mult_tool = MultiplyTool()
result = mult_tool.invoke({'a' : 3, 'b' : 5})

