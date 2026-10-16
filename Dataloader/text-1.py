from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
parser = StrOutputParser()
loader = TextLoader('poem.txt', encoding='utf-8')
doc = loader.load()

print(type(doc))
print(len(doc))
print(type(doc[0]))
print(doc[0].metadata)
print(doc[0])

prompt1 = PromptTemplate(
       template = 'Summarize this : {poem}',
       input_variables= ['poem']
)

chain = prompt1 | model | parser

print(chain.invoke({'poem' : doc[0].page_content}) )

