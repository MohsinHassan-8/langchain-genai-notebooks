# maximum marginal relevance
from langchain_community.vectorstores import FAISS
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

embedding_model = GoogleGenerativeAIEmbeddings(model = 'gemini-embedding-001')

vec_store = FAISS.from_documents(
       documents=docs,
       embedding = embedding_model
)

retriever = vec_store.as_retriever(
       search_type = 'mmr',
       search_kwargs = {'k' : 2, 'lambda_mult' : 0.5} ) #0 : more diverse, 1 : more similar

query = 'what is langchain ?'

result = retriever.invoke(query)

for i, doc in enumerate(result):
       print(f'Result : {i+1}')
       print(f'Content : {doc.page_content}')





