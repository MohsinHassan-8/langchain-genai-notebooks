from langchain_text_splitters import CharacterTextSplitter

text = """Hey! My name is Mohsin"""

splitter = CharacterTextSplitter(
       chunk_size = 5,
       chunk_overlap = 2,
       separator = '' 
)

result = splitter.split_text(text)

print(result)

