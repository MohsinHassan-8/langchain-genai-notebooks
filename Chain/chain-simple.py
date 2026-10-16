from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

parser = StrOutputParser()

temp1 = PromptTemplate(
       template = 'Give me summary about the {topic}',
       input_variables = ['topic']
)

temp2 = PromptTemplate(
       template = 'Give me important points from {text}',
       input_variables= ['text']
)

chain = temp1 | model | parser | temp2 | model | parser
result = chain.invoke({'topic' : 'bert'})
print(result)
chain.get_graph().print_ascii()

