from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()
url = 'https://www.daraz.pk/products/dress-shirt-for-men-summer-cotton-blend-formal-casual-slim-fit-long-sleeve-fashion-shirt-collection-multiple-color-i421784030-s1994659685.html?pvid=bfbf3b02-cad3-49c6-a24c-6150a730e607&search=jfy&scm=1007.51705.413671.0&spm=a2a0e.tm80335142.just4u.d_421784030'
model = ChatGoogleGenerativeAI( model = 'gemini-2.5-flash' )
parser = StrOutputParser()

loader = WebBaseLoader(url)

docs = loader.load()
print(len(docs))

print(docs[0].page_content)
print(docs[0].metadata)
