#!/usr/bin/env python3
"""Test script for MCP WhatsApp list_chats tool"""

import json
import subprocess
import sys

def test_list_chats():
    """Test the list_chats tool from WhatsApp MCP server"""
    try:
        # MCP tool call structure
        mcp_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "list_chats",
                "arguments": {}
            }
        }
        
        print("Testing MCP WhatsApp list_chats tool...")
        print(f"Request: {json.dumps(mcp_request, indent=2)}")
        
        # Note: Replace 'whatsapp-mcp-server' with actual server command
        result = subprocess.run(
            ["whatsapp-mcp-server"],
            input=json.dumps(mcp_request),
            text=True,
            capture_output=True,
            timeout=30
        )
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            print(f"Success: {json.dumps(response, indent=2)}")
            return response
        else:
            print(f"Error: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("Timeout: MCP server didn't respond within 30 seconds")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
    except FileNotFoundError:
        print("WhatsApp MCP server not found. Install or check path.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_list_chats()