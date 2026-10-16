from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(model = 'gemini-embedding-001')
model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

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

retriever = ContextualCompressionRetriever(
       base_retriever=vec_store.as_retriever(search_kwargs = {'k' : 2}),
       base_compressor=LLMChainExtractor.from_llm(model)
)

query = 'Any info about embeddings'

result = retriever.invoke(query)

for i,doc in enumerate(result):
       print(f'Result : {i+1}')
       print(f'Content : {doc.page_content}')

