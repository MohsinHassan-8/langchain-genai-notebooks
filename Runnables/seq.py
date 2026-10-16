from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
parser = StrOutputParser()

prompt1 = PromptTemplate(
       template='Tell me a joke (concise)'
)

chain = RunnableSequence(prompt1, model, parser)

result = chain.invoke({})
print(result)
