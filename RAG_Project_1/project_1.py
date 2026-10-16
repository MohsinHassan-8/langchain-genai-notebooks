from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

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

question = 'Is the topic of alien discussed in this video ? if yes, then what was discussed'

retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({'context' : context_text, 'question' : question})

#Generation
answer = model.invoke(final_prompt)
print(answer.content)

