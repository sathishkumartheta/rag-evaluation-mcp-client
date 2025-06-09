import os
import asyncio
import gradio as gr
import ast
from dotenv import load_dotenv
from mcp_playground import MCPClient, OpenAIBridge

load_dotenv()

# === MCP + OpenAI Setup ===
MCP_SERVER_URL = "https://eb43a6224df86401d9.gradio.live/gradio_api/mcp/sse"
OPENAI_MODEL = "gpt-4o"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

client = MCPClient(MCP_SERVER_URL)
bridge = OpenAIBridge(client, api_key=OPENAI_KEY, model=OPENAI_MODEL)

# === Async processing logic ===
async def run_query_async(prompt: str) -> str:
    try:
        result = await bridge.process_query(prompt)

        if result.get("tool_call"):
            tool = result["tool_call"]["name"]
            raw_output = result["tool_result"].content

            try:
                # Clean and parse the Python-dict-style string
                parsed = ast.literal_eval(raw_output.replace("root=", ""))
                scores = parsed.get("results", [])

                if not scores:
                    return f"✅ **Tool Used:** `{tool}`\n\n⚠️ No relevance scores returned."

                table = "\n".join([
                    f"- **Doc {i+1}** — Score: `{r['score']}`\n  > {r['document']}"
                    for i, r in enumerate(scores)
                ])

                return f"✅ **Tool Used:** `{tool}`\n\n📊 **Relevance Scores:**\n\n{table}"

            except Exception as parse_err:
                return f"✅ Tool: `{tool}`\n\n📦 Raw Output:\n```\n{raw_output}\n```"

        else:
            return f"🤖 GPT-4o Response (no tool used):\n\n{result['response'].content}"

    except Exception as e:
        return f"❌ Error: {str(e)}"

# Sync wrapper for Gradio
def run_query(prompt: str) -> str:
    return asyncio.run(run_query_async(prompt))

# === Gradio App ===
gr.Interface(
    fn=run_query,
    inputs=gr.Textbox(lines=12, label="Enter your query + documents"),
    outputs=gr.Markdown(label="Response"),
    title="🤖 RAG Evaluation Agent (OpenAI + MCP Tools)",
    description="This agent uses GPT-4o to evaluate your documents using tools like BM25 via MCP.",
    examples=[
        ["Evaluate the relevance of the following documents to the query:\n\nQuery: What are the benefits of drinking green tea?\n\nDocuments:\n1. Green tea contains antioxidants that help reduce inflammation.\n2. It is commonly consumed in East Asia and has cultural significance.\n3. Studies show green tea may aid in weight loss and improve brain function.\n4. Black tea is made from fermented leaves and contains more caffeine."]
    ]
).launch(share=True)
