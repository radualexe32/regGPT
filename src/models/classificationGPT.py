import os
import openai
from flask import Flask
from typing import List, Dict
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    MessagesPlaceholder, ChatPromptTemplate, BaseChatPromptTemplate
)
from pydantic import BaseModel, Field
from regGPT import Regression
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class ClassifierObjectJSON(BaseModel):
    # dataclass: Dict[str, str] = Field(
    #     description="Dictionary of variable name and classification type pairings."
    # )
    ind_var: str = Field(
        alias="Independent Variable", description="Classification of the independent variable")
    dep_var: str = Field(
        alias="Dependent Variable", description="Classification of the dependent variable")
    reg_type: str = Field(
        alias="Regression Model", description="Regression type to use"
    )
    deg_range: str = Field(
        alias="Degree Range", description="Range of degrees to use for polynomial regression"
    )


def classifier(data=[], correlation=None, reg_types=[]):
    template_file_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'templates', 'classification_template.txt')

    with open(template_file_path, 'r') as file:
        template_string = file.read()

    parser = PydanticOutputParser(pydantic_object=ClassifierObjectJSON)

    # Specify the model
    model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
    prompt = ChatPromptTemplate.from_template(template=template_string)
    response = prompt.format_messages(
        data=data,
        correlation=correlation,
        reg_types=reg_types
    )

    output = model(response)
    return parser.parse(output.content)


# def call_reg():
#    raise NotImplementedError
#
#
# def call_stat():
#    raise NotImplementedError


# app = Flask(__name__)


# @app.route('/')
# def index():
#     return '<p>Hello world!</p>'


# def main():
#     # Test data
#     data = [["value of average housing price in million", "percentage change in household income"], [0, 0], [0.5, 0.25], [1, 1], [1.5, 2.25], [2, 4], [2.5, 6.25], [3, 9], [
#         3.5, 12.25], [4, 16], [4.5, 20.25], [5, 25], [5.5, 30.25], [6, 36], [6.5, 42.25], [7, 49], [7.5, 56.25], [8, 64], [8.5, 72.25], [9, 81], [9.5, 90.25], [10, 100]]

#     data2 = [["type of drug treament (plabebo or vit C)", "occurence of common cold"], [0, 0.2], [1, 0.8], [0, 0.1], [1, 0.9], [0, 0.3], [1, 0.7], [0, 0.4], [1, 0.6], [0, 0.15], [1, 0.85], [
#         0, 0.25], [1, 0.75], [0, 0.35], [1, 0.65], [0, 0.05], [1, 0.95], [0, 0.1], [1, 0.9], [0, 0.2], [1, 0.8]]

#     correlation = 0
#     correlation2 = 0.936
#     reg_types = ["Linear regression",
#                  "Polynomial regression", "Logistic regression"]
#     out = classifier(data=data, correlation=correlation, reg_types=reg_types)
#     print(out)

#     out2 = classifier(data=data2, correlation=correlation2,
#                       reg_types=reg_types)
#     print(out2)


# if __name__ == "__main__":
#     main()
