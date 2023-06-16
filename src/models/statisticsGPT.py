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
from regGPT import Regression
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class StatisticsGPT(BaseModel):
    htests: Dict[str, List[str]] = Field(
        title="Statistics Tests",
        description="Dictionary for hypothesis test and prediction analysis tools."
    )


def inference_generator(variables={}, classfication={}, stats={}, reg_model=Regression(), degree=None, relation=None):
    # llm = OpenAI()
    # chat_model = ChatOpenAI(llm)
    template_file_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'templates', 'statistics_template.txt')

    with open(template_file_path, 'r') as file:
        template_string = file.read()

    prompt = ChatPromptTemplate.from_template(template=template_string)
    response = prompt.format_messages(
        variable_ind=None,
        classification_ind=None,
        variable_dep=None,
        classification_dep=None,
        coeff_line=None,
        MSE=None,
        R2=None,
        correlation=None,
        degree=degree,
        relationship=relation
    )


app = Flask(__name__)


@app.route('/')
def index():
    return '<p>Hello world!</p>'


def main():
    raise NotImplementedError


if __name__ == "__main__":
    inference_generator()
