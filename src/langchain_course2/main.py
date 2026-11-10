from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def main() -> None:
    # print("Hello from langchain-course2!")
    info = """Shaheer Sheikh is an Indian lawyer turned actor and model. He is best known for his role of Warrior prince Arjuna in mythological show Mahabharat (2013) and Dev Dixit in Kuch Rang Pyar Ke Aise Bhi (2016). He made his television debut with Disney Channel's series Kya Mast Hai Life as Vir Mehra. He made his debut in Indian daily soap opera as a lead actor with the role of Anant Bajpai in Navya: Naye Dhadkan Naye Sawaal (2011). He also portrayed historical roles like Nana Sahib in the show Jhansi Ki Rani and Salim in Dastaan-E-Mohabbat: Salim Anarkali.
Sheikh made his Hindi film debut in 2024, with the Netflix film Do Patti (2024) opposite Kriti Sanon.
Sheikh is also popular star in Indonesia. Due to his popularity in Indonesia, he has also starred in the Indonesian film Turis Romantis and the television series, Cinta Di Langit Taj Mahal. He has also hosted Asia's Got Talent and Indonesian reality show, Panah Asmara Arjuna. He also acted in the Indonesian television show, Roro Jonggrang.
    """
    
    summery = """
    given the information {info} about a person I want you to create:
    1. a short summary
    2. two interesting facts about them
    """

    summery_prompt = PromptTemplate(
        input_variables= ["info"], template= summery
    )

    # llm = ChatOpenAI(temperature=0, model="gpt-5")
    llm = ChatOllama(temperature=0, model="gemma:2b")
    chain = summery_prompt | llm
    res = chain.invoke(input = {"info":info})

    print(res.content)

if __name__=="__main__":
    main()