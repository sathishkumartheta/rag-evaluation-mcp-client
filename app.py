import gradio as gr
import os
import asyncio
from dotenv import load_dotenv
from mcp_playground import MCPClient
from smolagents import CodeAgent
from smolagents.adapters.mcp import MCPAdaptTool

load_dotenv()

# MCP Server connection
MCP_SERVER_URL = "https://785e87c0901f815632.gradio.live/gradio_api/mcp/sse"
client = MCPClient(MCP_SERVER_URL)

# Prompt builder
def make_prompt(query, documents, task_instruction):
    doc_list = "\n".join(f"{i+1}. {doc.strip()}" for i, doc in enumerate(documents.split("\n")) if doc.strip())
    return f"""
You are an advanced retrieval evaluation agent with access to multiple tools via the MCP server.

Use the following instruction to determine which tools to apply:
🧾 **Instruction**: {task_instruction.strip()}

Available tools:
1. 🧠 `bm25_relevance_scorer` — Lexical relevance.
2. 🤝 `semantic_relevance_scorer` — Semantic relevance.
3. 🔁 `redundancy_checker` — Redundancy or repetition.
4. ✅ `exact_match_checker` — Exact textual match.

---

📌 **Query**: "{query.strip()}"

📄 **Documents**:
{doc_list}

🔍 You may use one or more tools to fully satisfy the instruction. Include rationale behind tool selection and show results.
"""

# Async runner using SmolAgent
async def run_eval(query, documents, task_instruction):
    message = make_prompt(query, documents, task_instruction)
    raw_tools = await client.list_tools()
    tools = [MCPAdaptTool(tool, client=client) for tool in raw_tools]  # ✅ Wrap each MCP tool

    agent = CodeAgent(
        tools=tools,
        model={
            "provider": "openai",
            "model": "gpt-4o",
            "api_key": os.environ.get("OPENAI_API_KEY")
        }
    )

    result = await agent.run(message)

    if result.tool_result:
        return f"✅ Tool Used: {result.tool_call.name}\n\n📊 Result:\n{result.tool_result.content}"
    else:
        return f"🤖 No tool was called.\n\nLLM Response:\n{result.response.content}"

# Sync Gradio wrapper
def evaluate(query, documents, task_instruction):
    return asyncio.run(run_eval(query, documents, task_instruction))

# Async Gradio-compatible tool listing
async def list_tools():
    tools = await client.list_tools()
    if not tools:
        return "⚠️ No tools available or MCP server not reachable."
    return "🧰 Available Tools:\n" + "\n".join(f"- {tool.name}" for tool in tools)

# Gradio UI
with gr.Blocks(title="RAG Evaluation MCP Client") as iface:
    gr.Markdown("## 🔍 RAG Evaluation Agent (SmolAgent-powered)")
    gr.Markdown("Provide a query, documents, and an instruction. The agent will select and call tools via MCP.")

    with gr.Row():
        query = gr.Textbox(label="Query", placeholder="e.g., What are the benefits of drinking green tea?")
        instruction = gr.Textbox(label="Task Instruction", placeholder="e.g., Evaluate for redundancy")

    documents = gr.Textbox(label="Documents", placeholder="Enter one document per line", lines=10)
    output = gr.Textbox(label="Agent Response", lines=20)

    with gr.Row():
        eval_btn = gr.Button("🔍 Evaluate")
        list_btn = gr.Button("🧰 List Tools")

    eval_btn.click(fn=evaluate, inputs=[query, documents, instruction], outputs=output)
    list_btn.click(fn=list_tools, inputs=[], outputs=output)

if __name__ == "__main__":
    iface.launch(share=True)
