from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
#error
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about topic'),
]

# Create parser from schemas
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 facts about {topic}\n{format_instruction}',
    input_variables=['topic'],
    partial_variables={
        'format_instruction': parser.get_format_instructions()
    }
)

chain = template | model

result = chain.invoke({'topic': 'Black Hole'})

print(result)