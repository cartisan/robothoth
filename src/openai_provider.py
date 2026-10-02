import pprint
from typing import cast

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import ResponseInputParam, ToolParam
from openai.types.responses.response_input_param import ResponseInputItemParam

from src.tools import dispatch

load_dotenv()

VAULT_HOME = "/Users/leonid/code/robothoth/tests/test_vault"

tools: list[ToolParam] = [
    {
        "type": "function",
        "name": "list_notes",
        "description": """Return all notes as sorted vault-relative Markdown paths
under an optional directory.

The directory is interpreted relative to ``vault_path`` and searched
recursively. Only regular Markdown files inside the vault are returned.

Raises:
    FileNotFoundError: If the vault or requested directory does not exist.
    ValueError: If ``path`` is absolute, escapes the vault, or is not a
        directory inside the vault.""",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "An optional path relative to the vault home"
                                   "for which to return notes.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        "strict": None,
    },
]

system_prompt = """You are an assistant that helps navigate an Obsidian note vault.
The vault is located under a vault path on the local machine,
and is managed by your harness.
You only need to operate on file paths relative to that vault path.

Help the user with the following query:
"""

user_prompt = "Help me locate the file called 'Agentic Software Engineering Factory'."

client = OpenAI()

input_list: ResponseInputParam = [
    {"role": "developer", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]
response = client.responses.create(
    model="gpt-6-luna",
    tools=tools,
    input=input_list,
)

print(f"Tokens spent: {response.usage.total_tokens if response.usage else 'unknown'}")

# Save model's response in context
input_list.extend(
    cast(ResponseInputItemParam, item.to_dict()) for item in response.output
)

for item in response.output:
    if item.type == "function_call":
        tool_result = dispatch.dispatch_tool_call(
            name=item.name,
            arguments_json=item.arguments,
            vault_path=VAULT_HOME,
        )
        input_list.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": tool_result,
            }
        )

print("Final input:")
pprint.pprint(input_list)

response = client.responses.create(
    model="gpt-6-luna",
    tools=tools,
    input=input_list,
)
print(f"Tokens spent: {response.usage.total_tokens if response.usage else 'unknown'}")
print("Final output:")
print("\n" + response.output_text)
