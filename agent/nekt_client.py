import json
import requests


class NektClient:
    def __init__(self, url: str, token: str):
        self.url = url
        self.token = token
        self._session_id: str | None = None

    def _base_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

    def _parse_sse(self, text: str) -> dict | None:
        for line in text.splitlines():
            if line.startswith("data: "):
                return json.loads(line[6:])
        return None

    def initialize(self) -> str:
        resp = requests.post(
            self.url,
            headers=self._base_headers(),
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "seazone-data-agent", "version": "1.0"},
                },
                "id": "init",
            },
            timeout=30,
        )
        resp.raise_for_status()
        self._session_id = resp.headers.get("mcp-session-id")
        return self._session_id

    def call_tool(self, name: str, arguments: dict) -> dict:
        if not self._session_id:
            self.initialize()

        h = self._base_headers()
        h["mcp-session-id"] = self._session_id

        resp = requests.post(
            self.url,
            headers=h,
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
                "id": "call",
            },
            timeout=60,
        )
        resp.raise_for_status()

        data = self._parse_sse(resp.text)
        if not data:
            raise RuntimeError(f"Resposta inesperada: {resp.text[:500]}")

        if "error" in data:
            raise RuntimeError(f"Nekt error: {data['error']}")

        result = data.get("result", {})

        if "structuredContent" in result:
            return result["structuredContent"]

        content = result.get("content", [])
        if content and content[0].get("type") == "text":
            return json.loads(content[0]["text"])

        raise RuntimeError(f"Formato de resposta não reconhecido: {result}")

    def generate_sql(self, question: str, tables: list[str] | None = None) -> dict:
        args: dict = {"question": question}
        if tables:
            args["selected_tables"] = tables
        return self.call_tool("generate_sql", args)

    def execute_sql(self, sql: str) -> dict:
        return self.call_tool("execute_sql", {"sql_query": sql})

    def get_relevant_tables(self, question: str) -> list[str]:
        """Retorna lista de nomes de tabelas relevantes para a pergunta."""
        result = self.call_tool("get_relevant_tables_ddl", {"question": question})
        ddls = result.get("ddls", [])
        tables = []
        import re as _re
        for ddl in ddls:
            if "CREATE TABLE" not in ddl:
                continue
            # Formato com aspas: "db"."table"
            m = _re.search(r'"([^"]+)"\."([^"]+)"', ddl)
            if m:
                tables.append(m.group(2))
                continue
            # Formato sem aspas: db.table
            m = _re.search(r'CREATE TABLE\s+(\w+)\.(\w+)', ddl)
            if m:
                tables.append(m.group(2))
        return tables if tables else []
