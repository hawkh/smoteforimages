#!/usr/bin/env python3
"""Test MCP WhatsApp list_chats using mcp client library"""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_whatsapp_list_chats():
    """Test list_chats tool using MCP client"""
    
    # Server parameters - adjust path as needed
    server_params = StdioServerParameters(
        command="whatsapp-mcp-server",
        args=[]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
            
            # Call list_chats tool
            try:
                result = await session.call_tool("list_chats", {})
                print(f"\nlist_chats result:")
                print(result.content)
                return result
            except Exception as e:
                print(f"Error calling list_chats: {e}")

if __name__ == "__main__":
    asyncio.run(test_whatsapp_list_chats())