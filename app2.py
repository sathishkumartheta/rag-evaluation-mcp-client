import gradio as gr
import os
import asyncio
from dotenv import load_dotenv
from mcp_playground import MCPClient, OpenAIBridge

load_dotenv()

# MCP Server connection
MCP_SERVER_URL = "https://ecb3fb0f503b7d47f5.gradio.live/gradio_api/mcp/sse"
client = MCPClient(MCP_SERVER_URL)
bridge = OpenAIBridge(client, api_key=os.environ.get("OPENAI_API_KEY"), model="gpt-4o")

# Prompt builder
def make_prompt(query, documents, task_instruction):
    # Proper JSON-like array for the 'docs' argument
    doc_array = "[" + ",\n".join(f'"{doc.strip()}"' for doc in documents.split("\n") if doc.strip()) + "]"
    return f"""
You are an advanced retrieval evaluation agent with access to multiple tools via the MCP server.

Use the following instruction to determine which tools to apply:
🧾 Instruction: {task_instruction.strip()}

Available tools (you must format inputs exactly using argument names):

1. 🧠 `bm25_relevance_scorer(query: str, documents: List[str])`
2. 🤝 `semantic_relevance_scorer(query: str, documents: List[str])`
3. 🔁 `redundancy_checker(docs: List[str])`
4. ✅ `exact_match_checker(query: str, documents: List[str])`

---

📌 query = "{query.strip()}"

📄 documents (pass this to relevant tool as 'docs' or 'documents' argument):
{doc_array}

📣 Be sure to call the tool by passing named arguments only.
"""

async def run_eval(query, documents, task_instruction):
    message = make_prompt(query, documents, task_instruction)
    result = await bridge.process_query(message)

    # 🩹 HOTFIX: Inject docs if tool is redundancy_checker
    if result.get("tool_call"):
        tool = result["tool_call"]["name"]
        args = result["tool_call"].get("args", {})

        if tool == "redundancy_checker" and not args.get("docs"):
            docs_list = [doc.strip() for doc in documents.split("\n") if doc.strip()]
            result["tool_call"]["args"] = {"docs": docs_list}
            result["tool_result"] = await client.invoke(tool, **result["tool_call"]["args"])

        content = result["tool_result"].content
        return f"✅ Tool Used: {tool}\n\n📊 Result:\n{content}"
    else:
        return f"🤖 No tool was called.\n\nLLM Response:\n{result['response'].content}"


# Wrapper for evaluation
def evaluate(query, documents, task_instruction):
    return asyncio.run(run_eval(query, documents, task_instruction))

# Tool listing
async def list_tools():
    tools = await client.list_tools()
    if not tools:
        return "⚠️ No tools available or MCP server not reachable."
    return "🧰 Available Tools:\n" + "\n".join(f"- {tool.name}" for tool in tools)

# Gradio UI
with gr.Blocks(title="RAG Evaluation MCP Client") as iface:
    gr.Markdown("## 🔍 RAG Evaluation Agent (Instruction-Guided)")
    gr.Markdown("Provide a query, documents, and an instruction. The agent will select tools accordingly.")

    with gr.Row():
        query = gr.Textbox(label="Query", placeholder="e.g., What are the health benefits of eating apples?")
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
