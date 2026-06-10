## Contributing
👍🎉 First off, thanks for taking the time to contribute! 🎉👍

Our goal is to make environmental impact estimates as accessible as nutrition labels. We welcome all ideas from any backgrounds including documentation improvements, specialized report parsers, domain-specific LCI schemas, and more. Feel free to open an issue to discuss and/or submit a PR. Here are some instructions to get you started extending the system:

### Adding a new parser
An important part of making the information retrieval robust is relying as little as possible on the LLM agents to parse
the retrieved information. To this end, we develop vendor specific report parsers that can parse environmental reports
etc. from their standard formats. New parsers can be added to `agents_for_lifecycle_inventory/agentic_lci/retrieval/reports/vendor` and listed in `__init__.py`.

### Adding a new tool
The central registry of tools is the MCP server (`agents_for_lifecycle_inventory/agentic_lci/mcp/server.py`). These are called by the Stakeholder agent
(`agents_for_lifecycle_inventory/agentic_lci/agents/stakeholder_agent.py`) via the LLM adapters that integrate with the MCP server (`agents_for_lifecycle_inventory/agentic_lci/llm_adapters`). To add a new tool, all you have to do is add it as a function annotated with `@mcp.tool()` in the MCP server and make sure it has a descriptive docstring (as that and the name + function signature is what the Stakeholder agent uses to decide which tools to call). If it only requires a few lines of code, feel free to add it in-place. If it is more complicated, you can add utility functions in the appropriate subfolder(s) of `agents_for_lifecycle_inventory/agentic_lci/retrieval`.

### Adding a new schema
The self-play pipeline is modular and supports producing any LCI schema as relevant to the application at hand.
To add a new schema, add it as a pydantic model in a new file in `agents_for_lifecycle_inventory/agentic_lci/schemas` and list it with a description
in `__init__.py`. 

### Adding a new LLM provider
Whether it is a proprietary API or support for running local models, new LLM providers are always welcome.
Ideally the LLM(s) support text+image input, structured output, and tool calling (to hook up with the MCP server in `agents_for_lifecycle_inventory/agentic_lci/mcp`). If the provider has native tool support, e.g., for web-search, feel free to add that. The MCP also has
some fall-back tools for web-search for models that don't support it natively. To get started, add a new `*_adapter.py` file to `agents_for_lifecycle_inventory/agentic_lci/llm_adapters`, list it in `__init__.py`, and add instructions for any necessary API keys under the `.env` section in `agents_for_lifecycle_inventory/README.md`. 
