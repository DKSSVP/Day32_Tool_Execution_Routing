import json
import os

from openai import OpenAI
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY not found. Check your .env file."
    )


# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)


# Local tool function
def calculate_total(quantity, unit_price):
    return quantity * unit_price


# Register available local tools
tools = {
    "calculate_total": calculate_total
}


# Define tools for the LLM
tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "calculate_total",
            "description": "Calculate the total price using quantity and unit price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "quantity": {
                        "type": "integer",
                        "description": "Number of items"
                    },
                    "unit_price": {
                        "type": "number",
                        "description": "Price per item"
                    }
                },
                "required": ["quantity", "unit_price"]
            }
        }
    }
]


def execute_tool(tool_call):
    """Execute an approved local tool using LLM-provided arguments."""

    function_name = tool_call.function.name

    try:
        arguments = json.loads(
            tool_call.function.arguments
        )

        print("\nFunction Name:", function_name)
        print("Arguments:", arguments)

        if function_name not in tools:
            return {"error": "Unknown function"}

        result = tools[function_name](**arguments)

        print("Tool Execution Result:", result)

        return result

    except json.JSONDecodeError:
        return {"error": "Invalid JSON arguments"}

    except Exception as error:
        return {"error": str(error)}


def main():

    messages = [
        {
            "role": "user",
            "content": "Calculate the total for 5 items, each costing 100."
        }
    ]

    print("Sending request to OpenRouter...")

    try:
        # Step 1: Send request to OpenRouter
        response = client.chat.completions.create(
            model="openrouter/auto",
            messages=messages,
            tools=tool_definitions,
            tool_choice="auto",
            max_tokens=300
        )

        assistant_message = response.choices[0].message

        # Step 2: Add assistant response to conversation
        messages.append(assistant_message)

        # Step 3: Check whether the LLM requested a tool
        if assistant_message.tool_calls:

            for tool_call in assistant_message.tool_calls:

                # Step 4: Execute local tool
                result = execute_tool(tool_call)

                # Step 5: Return tool output to LLM context
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

            # Step 6: Send tool result back to OpenRouter
            final_response = client.chat.completions.create(
                model="openrouter/auto",
                messages=messages,
                tools=tool_definitions,
                tool_choice="none",
                max_tokens=300
            )

            print("\nFinal LLM Response:")
            print(
                final_response.choices[0].message.content
            )

        else:
            print("\nLLM Response:")
            print(assistant_message.content)

            print(
                "\nNo function call was returned by the model."
            )

    except Exception as error:
        print("\nOpenRouter Request Failed:")
        print(error)


if __name__ == "__main__":
    main()