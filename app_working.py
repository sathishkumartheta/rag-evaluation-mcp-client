import gradio as gr
import os
import asyncio
from dotenv import load_dotenv
from mcp_playground import MCPClient, OpenAIBridge

load_dotenv()

# Initialize MCP client and bridge
MCP_SERVER_URL = "https://59d7dd5931ea957432.gradio.live/gradio_api/mcp/sse"
client = MCPClient(MCP_SERVER_URL)
bridge = OpenAIBridge(client, api_key=os.environ.get("OPENAI_API_KEY"), model="gpt-4o")

# Compositional prompt template with instruction
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

# Async query runner
async def run_eval(query, documents, task_instruction):
    message = make_prompt(query, documents, task_instruction)
    result = await bridge.process_query(message)

    if result.get("tool_call"):
        tool = result["tool_call"]["name"]
        content = result["tool_result"].content
        return f"✅ Tool Used: {tool}\n\n📊 Result:\n{content}"
    else:
        return f"🤖 No tool was called.\n\nLLM Response:\n{result['response'].content}"

# Wrapper for Gradio
def evaluate(query, documents, task_instruction):
    return asyncio.run(run_eval(query, documents, task_instruction))

# Gradio UI
iface = gr.Interface(
    fn=evaluate,
    inputs=[
        gr.Textbox(label="Query", placeholder="e.g., What are the health benefits of eating apples?"),
        gr.Textbox(label="Documents", placeholder="Enter one document per line", lines=10),
        gr.Textbox(label="Task Instruction", placeholder="e.g., Evaluate for redundancy"),
    ],
    outputs=gr.Textbox(label="Evaluation Result", lines=20),
    title="🔍 RAG Evaluation Agent (Instruction-Guided)",
    description="Instruct the agent on what to evaluate (redundancy, relevance, match). It will use appropriate MCP tools."
)

if __name__ == "__main__":
    iface.launch(share=True)
