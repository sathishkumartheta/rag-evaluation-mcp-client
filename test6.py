import asyncio
import os
from mcp_playground import MCPClient, OpenAIBridge
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

async def main():
    # Connect to your MCP server
    client = MCPClient("https://1f700a7c745593ccb7.gradio.live/gradio_api/mcp/sse")

    # Set up OpenAI-powered agent
    bridge = OpenAIBridge(
        client,
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o"
    )

    # === Prompt example for multi-tool evaluation ===
    message = """
You are an advanced retrieval evaluation agent, equipped with specialized tools to analyze information.

Your objective is to thoroughly evaluate the relevance, redundancy, and factual alignment of a given set of documents with respect to a query. To achieve this, you may use **any combination of the tools** provided by the MCP server. Each tool has a specific function:

1. 🧠 `bm25_relevance_scorer`: Measures lexical (token-based) relevance between a query and documents.
2. 🤝 `semantic_relevance_scorer`: Measures semantic similarity using dense embeddings.
3. 🔁 `redundancy_checker`: Identifies redundant or repetitive information across documents.
4. ✅ `exact_match_checker`: Verifies whether a document contains an exact textual match for the query.

---

### 💡 Your capabilities:
- You can **use multiple tools sequentially or in parallel** to answer each task thoroughly.
- You should **choose tools based on the intent of the task**, and include reasoning if needed.
- If you're unsure which tool to use, you may perform exploratory steps using more than one tool.

---

### 🧪 Task:

Evaluate the semantic correctnes of the following documents for the query:  
📌 *"What are the benefits of drinking green tea?"*

📄 **Documents:**
1. Green tea contains antioxidants that help reduce inflammation.
2. It is commonly consumed in East Asia and has cultural significance.
3. Studies show green tea may aid in weight loss and improve brain function.
4. Black tea is made from fermented leaves and contains more caffeine.

Your response should reflect the tools used and the rationale behind your evaluation.
"""


    # Process the query
    result = await bridge.process_query(message)

    # Print result
    if result.get("tool_call"):
        print(f"✅ Tool: {result['tool_call']['name']}")
        #print("📦 Tool Arguments:")
        #pprint(result["tool_call"]["arguments"])
        print("\n📊 Tool Result:")
        pprint(result["tool_result"].content)
    else:
        print("🤖 No tool was called.")
        print("Response:", result["response"].content)

# Run it
asyncio.run(main())
