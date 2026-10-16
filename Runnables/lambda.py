from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel,RunnablePassthrough, RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
parser = StrOutputParser()

def word_count(text) : 
       return len(text.split())

prompt1 = PromptTemplate(
       template = 'Tell me a joke about : {topic}',
       input_variables= ['topic']
)

prompt2 = PromptTemplate(
       template = 'Length of the joke : {text}',
       input_variables= ['text']
)

joke_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
       'joke' : RunnablePassthrough(),
       'word_count' : RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_chain, parallel_chain)

result = final_chain.invoke({'topic' : 'cricket'})

final_result = """ {} \n word count : {}""".format(result['joke'], result['word_count'])
print(final_result)

