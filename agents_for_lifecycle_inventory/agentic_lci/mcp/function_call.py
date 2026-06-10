# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - Thin wrapper of the MCP server into regular OpenAI tool-calling format

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

import json
from agentic_lci.mcp.server import mcp

TOOLS = {
    tool.name: {
        "spec": {
            "name": tool.name,
            "description": tool.description,
            "type": "function",
            "parameters": tool.parameters,
        },
        "fn": {
            "fn": tool.fn,
            "fn_metadata": tool.fn_metadata,
            "is_async": tool.is_async,
        },
    }
    for tool in mcp._tool_manager.list_tools()
}


def call_tool(name, id, arguments):
    tool = TOOLS[name]["fn"]
    fn = tool["fn"]
    fn_metadata = tool["fn_metadata"]
    assert not tool["is_async"], "async tool calls not supported yet"

    try:
        arguments_pre_parsed = fn_metadata.pre_parse_json(json.loads(arguments))
        arguments_parsed_model = fn_metadata.arg_model.model_validate(
            arguments_pre_parsed
        )
        arguments_parsed_dict = arguments_parsed_model.model_dump_one_level()

        result = fn(**arguments_parsed_dict)

        _, structured_content = fn_metadata.convert_result(result)
    except Exception as e:
        structured_content = {"result": "Tool call gave error status: " + str(e)}

    return {
        "type": "function_call_output",
        "call_id": id,
        "output": json.dumps(structured_content),
    }


if __name__ == "__main__":
    # example/test usage
    print(call_tool("lookup_db_info", "test123", '{"product_name": "iPhone 13"}'))
