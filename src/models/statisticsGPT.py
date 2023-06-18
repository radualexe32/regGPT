import sys
import re
import csv
import pandas as pd
import os
import openai
import json
import yaml
from getpass import getpass
from typing import Callable, List, Union, Any, Dict
from pydantic import BaseModel, Field, validator, root_validator
from langchain import PromptTemplate, SerpAPIWrapper, LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.document_loaders import (
    PyPDFLoader, CSVLoader, UnstructuredPowerPointLoader, UnstructuredURLLoader
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory, ConversationTokenBufferMemory
from langchain.agents import (
    AgentType, initialize_agent, Tool, AgentExecutor, LLMSingleActionAgent,
    AgentOutputParser, create_json_agent
)
from langchain.prompts import (
    MessagesPlaceholder, ChatPromptTemplate, BaseChatPromptTemplate
)
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.output_parsers import PydanticOutputParser
from langchain.tools import StructuredTool
from langchain.llms import OpenAI
from langchain.tools.requests.tool import RequestsGetTool, TextRequestsWrapper
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from langchain.requests import TextRequestsWrapper
from langchain.tools.json.tool import JsonSpec
from flask import Flask
from dotenv import load_dotenv
import gradio as gr
from regGPT import Regression
from train_regression import train
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class StatisticsGPT(BaseModel):
    # htests: Dict[str, List[str]] = Field(
    #    title="Statistics Tests",
    #    description="Dictionary for hypothesis test and prediction analysis tools."
    # )
    h_tests: List[str] = Field(
        alias="Hypothesis Tests", description="Hypothesis test tools to conduct")
    reg_model_tests: List[str] = Field(
        alias="Regression Model Tests", description="Tests to conduct on the regression model itself")
    further_reg: List[str] = Field(
        alias="Further Regression Suggestions", description="Further regression model suggestions to use")
    pred_tools: List[str] = Field(alias="Prediction Analysis",
                                  description="Prediction tools useful for the data inputted")


def inference_generator(variables={}, classfication={}, stats={}, reg_model=Regression(), degree=None, relation=None):
    # llm = OpenAI()
    # chat_model = ChatOpenAI(llm)
    template_file_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'templates', 'statistics_template.txt')

    with open(template_file_path, 'r') as file:
        template_string = file.read()

    parser = PydanticOutputParser(pydantic_object=StatisticsGPT)
    prompt = ChatPromptTemplate.from_template(template=template_string)

    # Specify the model
    model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
    response = prompt.format_messages(
        variable_ind=list(variables.keys())[0],
        classification_ind=variables[list(variables.keys())[0]],
        variable_dep=list(variables.keys())[1],
        classification_dep=variables[list(variables.keys())[1]],
        coeff_line=reg_model.get(),
        MSE=stats["MSE"],
        R2=stats["R2"],
        correlation=stats["correlation"],
        degree=degree,
        relationship=relation
    )
    output = model(response)
    return parser.parse(output.content)


# app = Flask(__name__)


# @app.route('/')
# def index():
#     return '<p>Hello world!</p>'


# def main():
#     # Dummy inputs
#     reg = train()
#     variables = {"percentage change in income": "continuous continuous",
#                  "housing prices in USD": "continuous continuous"}
#     stats = {"MSE": 32.469417572021484, "R2": 0.9964616956476273,
#              "correlation": 0.99822998046875}

#     out = inference_generator(
#         variables=variables, stats=stats, reg_model=reg, degree=1, relation="positive linears")
#     print(out)


# if __name__ == "__main__":
#     main()
