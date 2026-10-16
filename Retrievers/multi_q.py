from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(model = 'gemini-embedding-001')

docs = [
       Document(page_content='Langchain helps devs to build llm apps easily'),
       Document(page_content='chroma is vector db for optimized search'),
       Document(page_content='Embeddings convert text into high dim vectors'),
       Document(page_content='Gemini provides powerful embedding models')
]

vec_store = Chroma.from_documents(
       documents=docs,
       embedding= embedding_model,
       collection_name = 'sample_1'
)

retriever = MultiQueryRetriever.from_llm(
       retriever= vec_store.as_retriever(search_kwargs = {'k' : 5}),
       llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
)

query = 'Any info about embeddings'

result = retriever.invoke(query)

for i,doc in enumerate(result):
       print(f'Result : {i+1}')
       print(f'Content : {doc.page_content}')

'''Difference:
k means
Plain retriever  : final output count
Multi-Query retriever : per-sub-query fetch count'''
