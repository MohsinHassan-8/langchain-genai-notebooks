from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

class Person(BaseModel) : 
       name : str = Field(description="Name of person")
       age : int = Field(gt=18, description="Age of person")
       city : str = Field(description="Name of city")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Genrate name, age, city of some random {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={
        'format_instruction': parser.get_format_instructions()
    }
)

chain = template | model | parser

result = chain.invoke({'place': 'Pakistani'})

print(result)