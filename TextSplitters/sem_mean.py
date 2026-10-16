from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
model = GoogleGenerativeAIEmbeddings(model = 'gemini-embedding-001')
text = """Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass.

The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams."""

splitter = SemanticChunker(
       GoogleGenerativeAIEmbeddings(model = 'gemini-embedding-001'),
       breakpoint_threshold_type="standard_deviation",
       breakpoint_threshold_amount=1
)

chunks = splitter.create_documents([text])
print(chunks[0].page_content)
print(chunks[1].page_content)
print(len(chunks))

embedding_vector_1 = model.embed_query(chunks[0].page_content)    
embedding_vector_2 = model.embed_query(chunks[1].page_content)
print(f"Embedding dimension: {len(embedding_vector_1)}")
print(f"Embedding dimension: {len(embedding_vector_2)}")
print(f"First 10 dimensions: {embedding_vector_1[:10]}")
