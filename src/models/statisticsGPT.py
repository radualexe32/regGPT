from imports import *
from regGPT import Regression

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class StatisticsGPT(BaseModel):
    # htests: Dict[str, List[str]] = Field(
    #    title="Statistics Tests",
    #    description="Dictionary for hypothesis test and prediction analysis tools."
    # )
    h_tests: List[str] = Field(
        alias="Hypothesis Tests", description="Hypothesis test tools to conduct"
    )
    reg_model_tests: List[str] = Field(
        alias="Regression Model Tests",
        description="Tests to conduct on the regression model itself",
    )
    further_reg: List[str] = Field(
        alias="Further Regression Suggestions",
        description="Further regression model suggestions to use",
    )
    pred_tools: List[str] = Field(
        alias="Prediction Analysis",
        description="Prediction tools useful for the data inputted",
    )


def inference_generator(variables={}, reg_model=Regression()):
    # llm = OpenAI()
    # chat_model = ChatOpenAI(llm)
    template_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "templates",
        "statistics_template.txt",
    )

    with open(template_file_path, "r") as file:
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
        MSE=reg_model.get_mse(),
        R2=reg_model.get_r2(),
        correlation=reg_model.get_correlation(),
    )
    output = model(response)
    return parser.parse(output.content)
