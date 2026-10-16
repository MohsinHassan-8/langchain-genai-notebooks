from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

class Feedback(BaseModel) : 
       sentiment : Literal['positive', 'negative'] = Field(description='Sentiment of the feedback')

parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
       template= "Classify sentiment of feedback as positive or negative : \n{feedback} \n{format_instruction}",
       input_variables=['feedback'],
       partial_variables={'format_instruction' : parser2.get_format_instructions()}
)

prompt2 = PromptTemplate(
       template = 'write appropriate response to this positive feedback :  \n{feedback}',
       input_variables=['feedback']
)

prompt3 = PromptTemplate(
       template = 'write appropriate response to this negative feedback : \n{feedback}',
       input_variables=['feedback']
)

clf_chain = prompt1 | model | parser2

branch_chain = RunnableBranch(
       (lambda x: x.sentiment == 'positive' , prompt2 | model | parser1),
       (lambda x: x.sentiment == 'negative' , prompt3 | model | parser1),
       RunnableLambda(lambda x: "couldn't find the sentiment")
)

chain = clf_chain | branch_chain

result = chain.invoke({'feedback' : 'This is terrible smartphone'})
print(result)

chain.get_graph().print_ascii()
