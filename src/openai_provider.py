import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
print(os.environ.get("OPENAI_API_KEY"))

client = OpenAI()
response = client.responses.create(
    model="gpt-6-luna",
#    tools=tools,
    input="Say Hi, and nothing more.",
)

print(response.output)
