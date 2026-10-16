from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

class Review(TypedDict):
       summary : Annotated[str, 'Breif summary of the review']
       sentiment : Annotated[str, 'Return Sentiment of the review either negative, positive or neutral']

structured_model = model.with_structured_output(Review)
result = structured_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the looks outdated compared to other brands. Hoping for a software update to fix this.""")
print(result)
