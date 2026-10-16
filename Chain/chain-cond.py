from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

parser = StrOutputParser()

prompt1 = PromptTemplate(
       template= "Classify sentiment of feedback as pos or neg : {feedback}",
       input_variables=['feedback']
)

clf_chain = prompt1 | model | parser

result = clf_chain.invoke({'feedback' : 'This is terrible smartphone'})
print(result)


