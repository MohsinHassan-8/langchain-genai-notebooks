import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------
app = FastAPI(title="YouTube Video Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Serve /static/* (style.css, script.js) directly from the frontend folder
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


# ----------------------------------------------------------------------
# Shared LLM / embedding objects (created once)
# ----------------------------------------------------------------------
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embed = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
parser = StrOutputParser()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

prompt = PromptTemplate(
    template="""You are a helpful assistant.
    Answer only from provided transcript context,
    If context is insufficient, just say I Don't Know.

    {context}

    Question : {question}""",
    input_variables=["question", "context"],
)

# In-memory cache: {(video_id, k): retriever_chain}
_chain_cache: dict[tuple[str, int], RunnableLambda] = {}


def format_docs(ret_docs):
    return "\n\n".join(doc.page_content for doc in ret_docs)


def build_chain(video_id: str, k: int):
    """Fetch transcript, build FAISS index, return a callable RAG chain."""
    try:
        yt_api = YouTubeTranscriptApi()
        transcript_list = yt_api.fetch(video_id=video_id, languages=["en"])
        transcript = " ".join(chunk.text for chunk in transcript_list)
    except TranscriptsDisabled:
        return None
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not fetch transcript: {exc}")

    if not transcript.strip():
        return None

    chunks = splitter.create_documents([transcript])
    vec_store = FAISS.from_documents(documents=chunks, embedding=embed)
    ret = vec_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "lambda_mult": 0.5},
    )

    retrieval_chain = RunnableParallel(
        {
            "docs": ret,
            "context": ret | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    def run_with_sources(question: str):
        retrieved = retrieval_chain.invoke(question)
        answer = (prompt | model | parser).invoke(retrieved)
        return {"answer": answer, "sources": retrieved["docs"]}

    return RunnableLambda(run_with_sources)


def get_chain(video_id: str, k: int):
    key = (video_id, k)
    if key not in _chain_cache:
        chain = build_chain(video_id, k)
        if chain is None:
            return None
        _chain_cache[key] = chain
    return _chain_cache[key]


# ----------------------------------------------------------------------
# API schema
# ----------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    video_id: str
    k: int = 5


class SourceItem(BaseModel):
    content: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chain = get_chain(req.video_id.strip(), req.k)
    if chain is None:
        raise HTTPException(
            status_code=404,
            detail="Transcript is not available for this video. Try a different Video ID.",
        )

    result = chain.invoke(req.question)
    sources = [SourceItem(content=doc.page_content) for doc in result["sources"]]
    return AskResponse(answer=result["answer"], sources=sources)


@app.post("/api/reset")
def reset_video(video_id: str, k: int = 5):
    """Force a fresh transcript fetch / rebuild for a given video_id + k."""
    _chain_cache.pop((video_id.strip(), k), None)
    return {"status": "cleared"}
