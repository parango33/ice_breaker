import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from agents.linkedin_lookup_agent import linkedin_lookup_agent

from third_parties.linkedin import scrape_linkedin_profile


load_dotenv()

information="""
Juan is a dj artist in new york. he is also an investment banket in wallstreet. He likes the color red and plays ennis and football and likes pastas.
"""

if __name__ == '__main__':
    print("Hello Langchain...")

    linkedin_profile_url = linkedin_lookup_agent(text='Pablo Arango publicis devops')

    summary_template= """
    given the Linkedin information {information} about a perso from I want you to create:
    1. a short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate( input_variables=["information"] , template=summary_template)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)


    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)
    print(linkedin_data)

    print(chain.run(information=linkedin_data))
    




