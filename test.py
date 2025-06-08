import gradio as gr
import os

from mcp import StdioServerParameters
from smolagents import InferenceClientModel, CodeAgent, ToolCollection, MCPClient

# Replace this with your actual MCP Server URL
MCP_SERVER_URL = "https://bd842f1dcc8ac91bbe.gradio.live/gradio_api/mcp/sse"

try:
    # Initialize MCP Client with your server URL
    mcp_client = MCPClient(
        {
            "url": MCP_SERVER_URL,
            "transport": "sse"
            }
            )

    # Dynamically fetch all tools from the RAG Evaluation MCP Server
    tools = mcp_client.get_tools()

    # Use Hugging Face LLM to power your agent (requires valid token in env)
    model = InferenceClientModel(token=os.getenv("HUGGINGFACE_API_TOKEN"))

    # Create a CodeAgent that can reason and call tools
    agent = CodeAgent(tools=tools, model=model)

    # Gradio frontend for user interaction
    demo = gr.ChatInterface(
        fn=lambda message, history: str(agent.run(message)),
        type="messages",
        examples=[
            "Which document is most relevant to 'quantum computing breakthroughs'?",
            "Check redundancy among these retrieved passages",
            "Evaluate hallucination in this RAG answer: ...",
        ],
        title="RAG Evaluation Agent",
        description="This agent uses MCP Tools (e.g., BM25 relevance scorer) to evaluate RAG outputs.",
    )

    # Launch Gradio app
    demo.launch(share=True)

finally:
    mcp_client.disconnect()
