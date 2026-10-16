from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel,RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
parser = StrOutputParser()

prompt1 = PromptTemplate(
       template = 'Tell me about this : {topic}',
       input_variables= ['topic']
)

prompt2 = PromptTemplate(
       template = 'Summarize it into 5 lines : {text}',
       input_variables= ['text']
)

report_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
       (lambda x : len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
       RunnablePassthrough()
)

final_chain = RunnableSequence(report_chain, branch_chain)

result = final_chain.invoke({'topic' : 'BERT'})
print(result)

