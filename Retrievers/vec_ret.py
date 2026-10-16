from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()

docs = [
       Document(page_content='Langchain helps devs to build llm apps easily'),
       Document(page_content='chroma is vector db for optimized search'),
       Document(page_content='Embeddings convert text into high dim vectors'),
       Document(page_content='Gemini provides powerful embedding models')
]

embedding_model = GoogleGenerativeAIEmbeddings( model = 'gemini-embedding-001' )

vec_store = Chroma.from_documents(
       documents= docs,
       embedding = embedding_model,
       collection_name='my_collection'
)

retriever = vec_store.as_retriever(search_kwargs = {'k' : 2})

query = 'what is chroma used for'

results = retriever.invoke(query)

for i,doc in enumerate(results):
       print(f'Results : {i+1}')
       print(doc.page_content)

