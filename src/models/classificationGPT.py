import openai
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class ClassifierObjectJSON(BaseModel):
    # dataclass: Dict[str, str] = Field(
    #     description="Dictionary of variable name and classification type pairings."
    # )
    ind_var: str = Field(
        alias="Independent Variable",
        description="Classification of the independent variable",
    )
    dep_var: str = Field(
        alias="Dependent Variable",
        description="Classification of the dependent variable",
    )
    reg_type: str = Field(
        alias="Regression Model", description="Regression type to use"
    )
    deg_range: str = Field(
        alias="Degree Range",
        description="Range of degrees to use for polynomial regression",
    )


def classifier(data=[], correlation=None, reg_types=[], extra=""):
    template_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "templates",
        "classification_template.txt",
    )

    with open(template_file_path, "r") as file:
        template_string = file.read()

    parser = PydanticOutputParser(pydantic_object=ClassifierObjectJSON)

    # Specify the model
    model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
    prompt = ChatPromptTemplate.from_template(template=template_string)
    response = prompt.format_messages(
        data=data, correlation=correlation, reg_types=reg_types, extra=extra
    )

    output = model(response)
    return parser.parse(output.content)
