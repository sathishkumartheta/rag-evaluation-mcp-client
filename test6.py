import asyncio
import os
from mcp_playground import MCPClient, OpenAIBridge
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

async def main():
    # Connect to your MCP server
    client = MCPClient("https://fab81580f23894f875.gradio.live/gradio_api/mcp/sse")

    # Set up OpenAI-powered agent
    bridge = OpenAIBridge(
        client,
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o"
    )

    # === Prompt example for multi-tool evaluation ===
    message = """
You are a retrieval evaluation assistant. Use the appropriate tool from the MCP server to complete each task:

1️⃣ Use `bm25_relevance_scorer` or `semantic_relevance_scorer` to score relevance.

2️⃣ Use `redundancy_checker` to check for similar/repetitive passages.

3️⃣ Use `exact_match_checker` if a direct match with the query needs verification.

---

Task:

Evaluate the relevance and redundancy of the following documents for the query: "What are the benefits of drinking green tea?"

Documents:
1. Green tea contains antioxidants that help reduce inflammation.
2. It is commonly consumed in East Asia and has cultural significance.
3. Studies show green tea may aid in weight loss and improve brain function.
4. Black tea is made from fermented leaves and contains more caffeine.
"""

    # Process the query
    result = await bridge.process_query(message)

    # Print result
    if result.get("tool_call"):
        print(f"✅ Tool: {result['tool_call']['name']}")
        print("📦 Tool Arguments:")
        pprint(result["tool_call"]["args"])
        print("\n📊 Tool Result:")
        pprint(result["tool_result"].content)
    else:
        print("🤖 No tool was called.")
        print("Response:", result["response"].content)

# Run it
asyncio.run(main())
