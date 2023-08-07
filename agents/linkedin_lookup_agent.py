from tools.tools import get_profile_url

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

#Agente encargado de utilizar SerpAPI para conseguir el url del linkedin de la persona buscada.
# lo hace utilizando el tool

def linkedin_lookup_agent(text:str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    template = """given the full name of {name_of_person} I want you to get me a link to their linkedin profile page.
        Your answer should only contain a URL
    """

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func = get_profile_url,
            description="usefull for when you need the Linkedin page url" #agente decide si usar esta herramienta con base en la descripcion
        ),
    ]

    #Equip agent with tools and capabilites to perform the task. 
    # AgentType: which framework to decide which tool to use
    # Verbose: true -> agent is aware of every step. see reasoning process
    agent = initialize_agent(
        tools_for_agent,llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    prompt_template = PromptTemplate(
        input_variables=["name_of_person"], template=template
    )
    linkedin_username = agent.run(prompt_template.format_prompt(name_of_person=text))


    return linkedin_username
