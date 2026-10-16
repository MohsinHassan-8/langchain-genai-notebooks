from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
parser = StrOutputParser()

prompt1 = PromptTemplate(
       template = 'Tell me a joke about : {topic}',
       input_variables= ['topic']
)

prompt2 = PromptTemplate(
       template = 'explain me following joke : {text}',
       input_variables= ['text']
)

joke_chain = RunnableSequence(prompt1, model, parser)

chain = RunnableParallel({
       'joke' : RunnablePassthrough(),
       'explaination' : RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_chain, chain)

result = final_chain.invoke({'topic' : 'cricket'})
print(result)





