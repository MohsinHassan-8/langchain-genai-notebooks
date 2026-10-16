from langchain_community.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')


class MultiplyStruc(BaseModel):
       a : int = Field( description='First number to add' )
       b : int = Field( description='Second number to add' )

def multiply_func(a : int, b : int) -> int:
       return a * b

mult_tool = StructuredTool.from_function(
       func= multiply_func,
       name= 'multiply',
       description= 'multiply 2 numbers',
       args_schema= MultiplyStruc
)

result = mult_tool.invoke({'a' : 3, 'b' : 5})
print(result)


