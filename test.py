import gradio as gr
import os
import traceback
from types import SimpleNamespace
from dotenv import load_dotenv
from openai import OpenAI

from mcp import StdioServerParameters
from smolagents import CodeAgent, MCPClient

load_dotenv()

# ✅ OpenAI-compatible model class
class OpenAIModel:
    def __init__(self, api_key, model="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt, **kwargs):
        # Normalize prompt to string
        if isinstance(prompt, list):
            lines = []
            for m in prompt:
                if isinstance(m, dict) and "content" in m:
                    content = m["content"]
                    if isinstance(content, list):
                        content = " ".join(str(c) for c in content)
                    lines.append(str(content))
                elif isinstance(m, str):
                    lines.append(m)
            prompt = "\n".join(lines)
        elif not isinstance(prompt, str):
            prompt = str(prompt)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_tokens", 1024)
        )

        return SimpleNamespace(
            content=response.choices[0].message.content.strip(),
            token_usage=SimpleNamespace(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        )

    def run(self, prompt):
        return self.generate(prompt)

# ✅ MCP Server URL
MCP_SERVER_URL = "https://63ccba7860addf8c94.gradio.live/gradio_api/mcp/sse"

try:
    # 🔌 Connect to MCP server
    mcp_client = MCPClient({
        "url": MCP_SERVER_URL,
        "transport": "sse"
    })

    tools = mcp_client.get_tools()
    print("✅ Loaded tools from MCP server:")
    for tool in tools:
        print(f" - {tool.name}")

    # ✅ Wrap .call to support kwargs → positional
    def wrap_tool_calls_positional(tools):
        for tool in tools:
            original_call = tool.call

            def make_wrapped_call(_call):
                def wrapped(*args, **kwargs):
                    if kwargs and not args:
                        return _call(*kwargs.values())
                    return _call(*args)
                return wrapped

            tool.call = make_wrapped_call(original_call)

    wrap_tool_calls_positional(tools)

    # 🔐 Load OpenAI key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError("❌ OPENAI_API_KEY not found in environment.")

    # 🧠 OpenAI model
    model = OpenAIModel(
        api_key=openai_api_key,
        model="gpt-3.5-turbo"
    )

    # 🤖 Agent with patched tools
    agent = CodeAgent(tools=tools, model=model)

    def agent_response(message, history):
        try:
            print("📨 User message:", message)
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

    # 🎛️ Gradio UI
    demo = gr.ChatInterface(
        fn=agent_response,
        type="messages",
        title="🧠 RAG Evaluation Agent (OpenAI)",
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
