from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

parser = JsonOutputParser()

temp1 = PromptTemplate(
       template = "Give me age, name and city of some fictional person \n {format_instruction}",
       input_variables=[],
       partial_variables={"format_instruction" : parser.get_format_instructions()}
)

chain = temp1 | model | parser
result = chain.invoke({})
print(result)

