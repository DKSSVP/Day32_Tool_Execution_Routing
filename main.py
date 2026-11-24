import json

# Sample function-call response from an LLM
llm_response = '''
{
    "name": "calculate_total",
    "arguments": {
        "quantity": 5,
        "unit_price": 100
    }
}
'''

# Convert JSON string into a Python dictionary
data = json.loads(llm_response)

print("Parsed LLM Response:")
print(data)

# Extract function name and arguments
function_name = data["name"]
arguments = data["arguments"]

print("\nFunction Name:", function_name)
print("Arguments:", arguments)

print("Quantity:", arguments["quantity"])
print("Unit Price:", arguments["unit_price"])


# Define local tool function
def calculate_total(quantity, unit_price):
    return quantity * unit_price


# Register available tools
tools = {
    "calculate_total": calculate_total
}

print("\nRegistered Tools:", list(tools.keys()))


# Execute the selected registered function
try:
    if function_name in tools:
        result = tools[function_name](**arguments)
        print("\nTool Execution Result:", result)
    else:
        raise ValueError("Unknown function")

except Exception as e:
    result = {"error": str(e)}
    print("\nTool Execution Failed:", result)

# Prepare tool result for the LLM chat context
tool_message = {
    "role": "tool",
    "name": function_name,
    "content": str(result)
}

print("\nTool Message:")
print(tool_message)