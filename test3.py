import gradio as gr
import os
import traceback

from mcp import StdioServerParameters
from smolagents import InferenceClientModel, CodeAgent, MCPClient
from dotenv import load_dotenv
load_dotenv()

# MCP Server URL — update if changed
MCP_SERVER_URL = "https://a59b4424060a3dc4aa.gradio.live/gradio_api/mcp/sse"

try:
    # Initialize MCP Client
    mcp_client = MCPClient({
        "url": MCP_SERVER_URL,
        "transport": "sse"
    })

    # Fetch tools from the MCP Server
    tools = mcp_client.get_tools()

    print("✅ Loaded tools from MCP server:")
    for tool in tools:
        print(f" - {tool.name}")

    # Load model with Hugging Face API Token
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not token:
        raise EnvironmentError("❌ HUGGINGFACE_API_TOKEN not found in environment.")
    model = InferenceClientModel(token=token,model='mistralai/Mistral-7B-Instruct-v0.2')

    # Instantiate CodeAgent with fetched tools
    agent = CodeAgent(tools=tools, model=model)

    # Function that handles user input
    def agent_response(message, history):
        try:
            print("📨 User message:", message)

            # Inject light system hint (optional)
            message = (
                "You are a RAG Evaluation Agent. Use the appropriate MCP tools like bm25_relevance_scorer to score relevance, "
                "redundancy_checker for redundancy, or hallucination_checker for answer validation.\n\n"
                + message
            )

            result = agent.run(message)
            print("✅ Agent response:", result)
            return str(result)
        except Exception as e:
            print("🔥 Exception during agent.run():", e)
            traceback.print_exc()
            return f"❌ Error: {str(e)}"

    # Gradio UI
    demo = gr.ChatInterface(
        fn=agent_response,
        type="messages",
        title="🧠 RAG Evaluation Agent",
        description="LLM agent decides which MCP tools to use to evaluate RAG output. Just describe your setup!",
        examples=[
            "Evaluate the relevance of the following documents to the query 'What are the effects of turmeric on health?'\nDocuments:\n1. Turmeric contains curcumin, which has anti-inflammatory effects.\n2. Curcumin is an active ingredient that reduces joint pain.",
            "Check for redundancy in the following retrieved passages:\n1. Turmeric reduces inflammation.\n2. Turmeric is anti-inflammatory.\n3. It reduces swelling and inflammation.",
            "Use bm25_relevance_scorer with query 'What causes climate change?' and documents: 1. Greenhouse gases trap heat. 2. CO2 emissions have increased since the industrial era."
        ]
    )

    demo.launch(share=True)

finally:
    mcp_client.disconnect()
