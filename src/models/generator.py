import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class GeneratorParserJSON(BaseModel):
    topic: str = Field(
        alias="topic", description="The topic question the user has asked"
    )


def generator():
    pass


if __name__ == "__main__":
    print("this is the generator")
