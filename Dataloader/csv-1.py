from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader

load_dotenv()
loader = CSVLoader(file_path = 'p1.csv', encoding='utf-8')
model = ChatGoogleGenerativeAI( model = 'gemini-2.5-flash' )
parser = StrOutputParser()


docs = loader.load()
print(len(docs))

print(docs[0].page_content)
print(docs[0].metadata)
