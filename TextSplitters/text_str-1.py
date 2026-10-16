from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """Hey! My name is Mohsin
       I am currently unemployed
       
       And getting fucked up by AI"""

splitter = RecursiveCharacterTextSplitter(
       chunk_size = 25,
       chunk_overlap = 0
)

chunks = splitter.split_text(text)
print(chunks)
