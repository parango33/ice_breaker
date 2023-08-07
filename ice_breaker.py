import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain


load_dotenv()

information="""
Juan is a dj artist in new york. he is also an investment banket in wallstreet. He likes the color red and plays ennis and football and likes pastas.
"""

if __name__ == '__main__':
    print("Hello Langchain...")


    summary_template = """
    given the information {information} about a person from I want you to create:
    1. a short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate( input_variables=["information"] , template=summary_template)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    print(chain.run(information=information))




