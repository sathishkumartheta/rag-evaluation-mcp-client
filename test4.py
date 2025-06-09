import asyncio
import os
from mcp_playground import MCPClient, OpenAIBridge
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

async def main():
    # Connect to your MCP server
    client = MCPClient("https://d073d7a9000d02249b.gradio.live/gradio_api/mcp/sse")

    # Set up OpenAI-powered agent
    bridge = OpenAIBridge(
        client,
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4o"  # or "gpt-4", "gpt-3.5-turbo"
    )

    # Run the query
    message="""Evaluate the relevance of the following documents to the query:

            Query: What are the benefits of drinking green tea?

            Documents:
            1. Green tea contains antioxidants that help reduce inflammation.
            2. It is commonly consumed in East Asia and has cultural significance.
            3. Studies show green tea may aid in weight loss and improve brain function.
            4. Black tea is made from fermented leaves and contains more caffeine."""

    result = await bridge.process_query(message)

    # Check if a tool was used
    if result["tool_call"]:
        print(f"Tool: {result['tool_call']['name']}")
        print(f"Result: {result['tool_result'].content}")
        #pprint(result)
    else:
        print("No tool was called.")
        

# Run the async main function
asyncio.run(main())
