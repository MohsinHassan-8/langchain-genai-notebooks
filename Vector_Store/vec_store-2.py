from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document

doc1 = Document(
    page_content="""Islamabad United is one of the most successful PSL teams, having won the championship twice in 2016 and 2018. The team is known for its strong batting lineup and aggressive playing style. Shadab Khan currently leads the team as captain.""",
    metadata={"team": "Islamabad United", "founded": 2015, "championships": 2, "captain": "Shadab Khan", "city": "Islamabad", "color": "Red"}
)

doc2 = Document(
    page_content="""Peshawar Zalmi is a popular PSL franchise known for its passionate fan base and iconic yellow colors. The team won the PSL championship in 2017 under the captaincy of Darren Sammy. Babar Azam has been a key batsman for the team.""",
    metadata={"team": "Peshawar Zalmi", "founded": 2015, "championships": 1, "captain": "Babar Azam", "city": "Peshawar", "color": "Yellow"}
)

doc3 = Document(
    page_content="""Karachi Kings is a franchise representing Pakistan's largest city in the PSL. After struggling in the early seasons, they finally won their first championship in 2020. The team is currently led by Shan Masood and has a strong bowling attack.""",
    metadata={"team": "Karachi Kings", "founded": 2015, "championships": 1, "captain": "Shan Masood", "city": "Karachi", "color": "Blue"}
)

doc4 = Document(
    page_content="""Lahore Qalandars are known for their electric green jersey and one of the most loyal fan bases in the PSL. They won their first PSL championship in 2022 and defended it successfully in 2023. Shaheen Afridi is the charismatic captain of the team.""",
    metadata={"team": "Lahore Qalandars", "founded": 2015, "championships": 2, "captain": "Shaheen Afridi", "city": "Lahore", "color": "Green"}
)

doc5 = Document(
    page_content="""Multan Sultans are the newest addition to the PSL, having joined in 2018. They made an immediate impact and won their maiden PSL title in 2021. Muhammad Rizwan serves as the wicket-keeper and captain of the team.""",
    metadata={"team": "Multan Sultans", "founded": 2018, "championships": 1, "captain": "Muhammad Rizwan", "city": "Multan", "color": "Purple"}
)

doc6 = Document(
    page_content="""Quetta Gladiators are famous for their pink uniforms and never-give-up attitude on the field. They reached the finals in 2017 and 2019 but have yet to win a PSL championship. The team is captained by Sarfaraz Ahmed.""",
    metadata={"team": "Quetta Gladiators", "founded": 2015, "championships": 0, "captain": "Sarfaraz Ahmed", "city": "Quetta", "color": "Pink"}
)

psl_documents = [doc1, doc2, doc3, doc4, doc5, doc6]

vec_store = Chroma(
       embedding_function= GoogleGenerativeAIEmbeddings(model = 'gemini-embedding-001'),
       persist_directory='chroma_db',
       collection_name= 'sample'
)

vec_store.add_documents(psl_documents)

result = vec_store.similarity_search_with_score(
       query='Shaheen Afridi is captain of which team ? ',
       k = 2
)
print(result)
