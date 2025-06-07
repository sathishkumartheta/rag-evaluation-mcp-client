from smolagents.mcp_client import MCPClient



mcp_client=MCPClient(
    {
        "url": "https://0e1973235dcfafdf84.gradio.live/gradio_api/mcp/sse"
    }
)

tools=mcp_client.get_tools()
print(tools)