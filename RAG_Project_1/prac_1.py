from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
embed = GoogleGenerativeAIEmbeddings(model = 'gemini-embedding-001')
parser = StrOutputParser()
splitter = RecursiveCharacterTextSplitter(
       chunk_size = 1000,
       chunk_overlap = 200
)

video_id = 'LPZh9BOjkQs'
try:
       yt_api = YouTubeTranscriptApi()
       transcript_list = yt_api.fetch(video_id=video_id, languages=['en'])
       transcript = " ".join(chunk.text for chunk in transcript_list)
except TranscriptsDisabled:
       print('No transcript available')

chunks = splitter.create_documents([transcript])

vec_store = FAISS.from_documents(
       documents=chunks,
       embedding=embed
)

ret = vec_store.as_retriever(
       search_type = 'mmr',
       search_kwargs = {'k' : 5, 'lambda_mult' : 0.5}
)

def format_docs(ret_docs):
       context_text = "\n\n".join(docs.page_content for docs in ret_docs)
       return context_text


parallel_chain = RunnableParallel({
       'question' : RunnablePassthrough(),
       'context' : ret | RunnableLambda(format_docs)
})

prompt = PromptTemplate(
       template = '''You are a helpful assistant.
       Answer only from provided transcript context,
       If context is insufficient, just say You Don't Know.
       
       {context}
       Question : {question}''',
       input_variables=['context', 'question']
)

main_chain = parallel_chain | prompt | model | parser

ans = main_chain.invoke('Summarize')

print(ans)

