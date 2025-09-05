import json
import sys

def test_list_chats_mock():
    """Simple mock test for MCP list_chats tool"""
    
    # Mock MCP request
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "list_chats",
            "arguments": {}
        }
    }
    
    print("MCP WhatsApp list_chats Test")
    print("=" * 30)
    print(f"Request: {json.dumps(request, indent=2)}")
    
    # Mock response (what you'd expect from WhatsApp server)
    mock_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "chats": [
                            {"id": "chat1", "name": "Family Group", "type": "group"},
                            {"id": "chat2", "name": "John Doe", "type": "individual"},
                            {"id": "chat3", "name": "Work Team", "type": "group"}
                        ]
                    }, indent=2)
                }
            ]
        }
    }
    
    print(f"\nExpected Response: {json.dumps(mock_response, indent=2)}")
    return mock_response

if __name__ == "__main__":
    test_list_chats_mock()