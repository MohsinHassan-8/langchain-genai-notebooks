from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

load_dotenv()

model = ChatGoogleGenerativeAI( model = 'gemini-2.5-flash' )
parser = StrOutputParser()

loader = DirectoryLoader(
       path = 'eg_dir',
       glob = '*.pdf',
       loader_cls = PyPDFLoader
)

docs = loader.lazy_load()

for doc in docs:
       print(doc.metadata)

