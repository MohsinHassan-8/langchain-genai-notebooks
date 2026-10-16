from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
parser = StrOutputParser()

prompt1 = PromptTemplate(
       template='generate a tweet about {topic} (concise)',
       input_variables=['topic']
)

prompt2 = PromptTemplate(
       template='generate a linkedin post about {topic} (concise)',
       input_variables=['topic']
)

chain = RunnableParallel({
       'tweet' : RunnableSequence(prompt1, model, parser),
       'linkedin' : RunnableSequence(prompt2, model, parser)
       })

result = chain.invoke({'topic' : 'AI'})
print(result)
