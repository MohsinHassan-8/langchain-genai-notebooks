from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

#Doc Loader
video_id = 'LPZh9BOjkQs'
try:
       yt_api = YouTubeTranscriptApi()
       transcript_list = yt_api.fetch(video_id, languages=['en'])

       transcript = " ".join(chunk.text for chunk in transcript_list)
       print(transcript)
except TranscriptsDisabled:
       print('No captions available for this video')

#Text Splitter
splitter = RecursiveCharacterTextSplitter(
       chunk_size=1000,
       chunk_overlap = 200
)
chunks = splitter.create_documents([transcript])
len(chunks)

#Embeddings
embed = GoogleGenerativeAIEmbeddings(model = 'gemini-embedding-001')
vec_store = FAISS.from_documents(
       documents=chunks,
       embedding=embed
)

#print(vec_store.index_to_docstore_id)
#print(vec_store.get_by_ids('...'))

#Retrieval
retriever = vec_store.as_retriever(
       search_type = 'mmr',
       search_kwargs = {'k' : 4, 'lambda_mult' : 0.5}
)

#Augmentation
model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
prompt = PromptTemplate(
       template = '''You are a helpful assistant.
       Answer only from provided transcript context,
       If context is insufficient, just say You Don't Know.
       
       {context}
       Question : {question}''',
       input_variables=['context', 'question']
)

def format_docs(retrieved_docs):
       context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
       return context_text

parallel_chain = RunnableParallel({
       'question' : RunnablePassthrough(),
       'context' : retriever | RunnableLambda(format_docs)
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | model | parser

#Generation
answer = main_chain.invoke('Summarize the video')
print(answer)
