from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
model2 = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
parser = StrOutputParser()

temp1  = PromptTemplate(
       template = 'Make notes for this {topic}',
       input_variables= ['topic']
)

temp2 = PromptTemplate(
       template = 'make quiz for this {topic}',
       input_variables=['topic']
)

temp3 = PromptTemplate(
       template = 'Merge both of these into one : {notes} and {quiz}',
       input_variables=['notes', 'quiz']
)

parallel_chain = RunnableParallel({
       'notes' : temp1 | model1 | parser,
       'quiz' : temp2 | model2 | parser
})

merge_chain = temp3 | model1 | parser

chain = parallel_chain | merge_chain

result = chain.invoke({'topic' : 'bert'})

print(result)

chain.get_graph().print_ascii()

