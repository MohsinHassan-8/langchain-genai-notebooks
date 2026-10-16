from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
parser = StrOutputParser()
loader = PyPDFLoader('CH_7_Interview.pdf')
docs = loader.load()

print(type(docs))
print(len(docs))
print(type(docs[0]))
print(docs[0].metadata)
print(docs[1].page_content)

