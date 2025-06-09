import gradio as gr
import json
import os
from openai import OpenAI  # ✅ new-style import
from dotenv import load_dotenv
from smolagents import CodeAgent, MCPClient

# Load environment variables (for OpenAI API key)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # ✅ new client object

# Replace with your actual MCP server URL
MCP_SERVER_URL = "https://0062f3bbf9d0bc0ced.gradio.live/gradio_api/mcp/sse"

# Initialize MCP client
mcp_client = MCPClient(
    {
        "url": MCP_SERVER_URL,
        "transport": "sse"
    }
)

def list_tools():
    """
    Returns the list of tools available on the MCP server.
    """
    return json.dumps(mcp_client.list_tools(), indent=2)

def llm_decider(instruction: str, query: str, documents: str, generations: str):
    """
    Uses LLM to decide which tool(s) to call on the MCP server based on the given inputs.
    """
    tool_description_prompt = f'''
You are an intelligent AI agent tasked with selecting the best evaluation tool(s) for a given task.

Your MCP Server has the following tools available:
- BM25 Relevance Scorer
- Semantic Relevance Scorer
- Redundancy Checker
- Exact Match Checker
- Repetition Checker
- Semantic Diversity Checker
- Length Consistency Checker
- System Relevance Evaluator
- System Coverage Evaluator

Based on the user's instruction:
"{instruction}"

And inputs:
Query: {query}
Documents: {documents[:500]}...
Generations: {generations[:500]}...

Respond with a JSON list of tool names and arguments. Example:
[
  {{"tool": "BM25 Relevance Scorer", "args": {{"query": "...", "documents": "..."}}}},
  {{"tool": "System Relevance Evaluator", "args": {{"query": "...", "generations": "..."}}}}
]
'''

    # ✅ NEW SYNTAX (openai>=1.0.0)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": tool_description_prompt}],
        temperature=0.3
    )

    tool_calls = json.loads(response.choices[0].message.content)

    results = []
    for call in tool_calls:
        result = mcp_client.call_tool(call["tool"], call["args"])
        results.append({"tool": call["tool"], "result": result})

    return json.dumps(results, indent=2)

# Gradio interface
demo = gr.TabbedInterface(
    [
        gr.Interface(
            fn=llm_decider,
            inputs=[
                gr.Textbox(label="Instruction", lines=2),
                gr.Textbox(label="Query", lines=2),
                gr.Textbox(label="Documents", lines=6),
                gr.Textbox(label="Generations", lines=6)
            ],
            outputs=gr.Code(label="Tool Calls and Results (JSON)"),
            title="🧠 LLM Decision Agent"
        ),
        gr.Interface(
            fn=list_tools,
            inputs=[],
            outputs=gr.Code(label="Available Tools (JSON)"),
            title="🛠️ List Tools"
        )
    ],
    tab_names=["LLM Agent", "List Tools"]
)

demo.launch(share=True)
