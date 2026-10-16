from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

chat_template = ChatPromptTemplate([
       ('system', 'You are helpful {domain} expert'),
       ('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain' : 'cricket', 'topic' : 'swing'})

print(prompt)




