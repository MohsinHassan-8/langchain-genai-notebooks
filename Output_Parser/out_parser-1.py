from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
temp1 = PromptTemplate(
       template = "Write detailed report on {topic}",
       input_variables=["topic"]
)
temp2 = PromptTemplate(
       template="write 5 lines summary about \n {text}",
       input_variables=["text"]
)
prompt1 = temp1.invoke({"topic" : "black_hole"})
result1 = model.invoke(prompt1)
prompt2 = temp2.invoke({"text" : result1.content})
result2 = model.invoke(prompt2)
print(result2.content)



