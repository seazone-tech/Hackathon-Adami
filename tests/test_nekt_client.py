"""Tests for NektClient — HTTP client for Nekt MCP Server."""
from unittest.mock import MagicMock, patch


def test_initialize_returns_session_id():
    from agent.nekt_client import NektClient

    mock_resp = MagicMock()
    mock_resp.headers = {"mcp-session-id": "abc123"}
    mock_resp.text = 'event: message\ndata: {"jsonrpc":"2.0","id":"init","result":{}}'
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        client = NektClient("https://fake.url/mcp", "token")
        session = client.initialize()

    assert session == "abc123"
    assert client._session_id == "abc123"


def test_call_tool_returns_structured_content():
    import json
    from agent.nekt_client import NektClient

    client = NektClient("https://fake.url/mcp", "tok")
    client._session_id = "sess-1"

    structured = {"status": "succeeded", "columns": ["test"], "data": [["1"]]}
    mock_resp = MagicMock()
    mock_resp.text = f'event: message\ndata: {json.dumps({"jsonrpc":"2.0","id":"call","result":{"structuredContent": structured}})}'
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        result = client.call_tool("execute_sql", {"sql_query": "SELECT 1"})

    assert result == structured


def test_call_tool_raises_on_nekt_error():
    import json
    from agent.nekt_client import NektClient
    import pytest

    client = NektClient("https://fake.url/mcp", "tok")
    client._session_id = "s"

    error_payload = {"jsonrpc": "2.0", "id": "call", "error": {"code": -1, "message": "bad sql"}}
    mock_resp = MagicMock()
    mock_resp.text = f'event: message\ndata: {json.dumps(error_payload)}'
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="Nekt error"):
            client.call_tool("execute_sql", {"sql_query": "DROP TABLE x"})


def test_call_tool_auto_initializes_if_no_session():
    import json
    from agent.nekt_client import NektClient

    client = NektClient("https://fake.url/mcp", "tok")

    init_resp = MagicMock()
    init_resp.headers = {"mcp-session-id": "auto-sess"}
    init_resp.text = 'event: message\ndata: {"jsonrpc":"2.0","id":"init","result":{}}'
    init_resp.raise_for_status = MagicMock()

    structured = {"status": "succeeded", "columns": ["v"], "data": [["1"]]}
    call_resp = MagicMock()
    call_resp.text = f'event: message\ndata: {json.dumps({"jsonrpc":"2.0","id":"call","result":{"structuredContent": structured}})}'
    call_resp.raise_for_status = MagicMock()

    with patch("requests.post", side_effect=[init_resp, call_resp]):
        result = client.call_tool("execute_sql", {"sql_query": "SELECT 1"})

    assert client._session_id == "auto-sess"
    assert result == structured


def test_generate_sql_calls_nekt_tool():
    from agent.nekt_client import NektClient

    client = NektClient("https://fake.url/mcp", "tok")
    client._session_id = "s"

    gen_result = {"sql": "SELECT SUM(value) FROM t", "is_valid": True}
    with patch.object(client, "call_tool", return_value=gen_result) as mock_call:
        result = client.generate_sql("Quanto faturamos?", tables=["asaas_transactions"])

    mock_call.assert_called_once_with(
        "generate_sql",
        {"question": "Quanto faturamos?", "selected_tables": ["asaas_transactions"]},
    )
    assert result["sql"] == "SELECT SUM(value) FROM t"


def test_execute_sql_calls_nekt_tool():
    from agent.nekt_client import NektClient

    client = NektClient("https://fake.url/mcp", "tok")
    client._session_id = "s"

    exec_result = {"status": "succeeded", "columns": ["total"], "data": [["500000"]]}
    with patch.object(client, "call_tool", return_value=exec_result) as mock_call:
        result = client.execute_sql("SELECT SUM(value) FROM t")

    mock_call.assert_called_once_with("execute_sql", {"sql_query": "SELECT SUM(value) FROM t"})
    assert result["status"] == "succeeded"
