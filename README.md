# LangChain / GenAI Notebooks

A topic-by-topic LangChain learning repo, ending in two RAG applications. Main LLM: Gemini 2.5 Flash.

| Stage | Folder | Covers |
|---|---|---|
| 1 | `ChatModels` | Gemini, a Hugging Face endpoint, a local TinyLlama model |
| 2 | `Prompts` | Prompt templates, message placeholders with chat history, a CLI chatbot, a Streamlit paper summariser |
| 3 | `Output_Parser`, `Structured_Output` | JSON and Pydantic parsers; `with_structured_output` using TypedDict, Pydantic and JSON Schema |
| 4 | `Chain`, `Runnables` | Sequential, parallel and conditional chains; LCEL runnables |
| 5 | `Dataloader`, `TextSplitters` | PDF, CSV, web, text and directory loaders; character, recursive, code-aware and semantic splitters |
| 6 | `Vector_Store`, `Retrievers` | Chroma with metadata filtering; similarity, MMR, multi-query and compression retrievers |
| 7 | `Tools`, `Tool_Calling` | Custom tools, tool-calling loops, an agent, a two-tool currency converter |
| 8 | `Youtube_Project` | YouTube-transcript RAG with FAISS, FastAPI backend and a simple web frontend |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env     # then fill in your own keys
```

Keys used: `GOOGLE_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`, `HF_TOKEN`, and `EXCHANGE_RATE_API_KEY` (for the currency tool). Vector stores (`chroma_db/`) are generated on first run and are not committed.

---
**Author:** Mohsin Hassan · [GitHub](https://github.com/MohsinHassan-8) · [LinkedIn](https://linkedin.com/in/mohsin-hassan-gen-ai)
