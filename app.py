from smolagents.mcp_client import MCPClient



mcp_client=MCPClient(
    {
        "url": "https://bbb9d7df1202baf06e.gradio.live/gradio_api/mcp/sse",
        "transport" : "sse"
    }
)

tools=mcp_client.get_tools()
print("\n".join(f"{t.name}: {t.description}" for t in tools))