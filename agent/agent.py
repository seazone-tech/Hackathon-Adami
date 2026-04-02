import json
import anthropic
import pandas as pd
from agent.disambiguator import classify_with_tables
from agent.table_map import route_tables


INTERPRET_PROMPT = """Você é um analista de dados da Seazone. A Nekt já calculou os dados e forneceu uma explicação técnica. Seu trabalho é apresentar isso de forma clara e analítica ao usuário.

IMPORTANTE: Responda SEMPRE em português brasileiro. A explicação da Nekt está em inglês — traduza e reescreva para o usuário.

Pergunta original: {question}

Explicação da Nekt (em inglês, técnica — traduza): {nekt_explanation}

Colunas: {columns}
Dados (até 20 linhas): {data}

Responda em JSON com exatamente este formato:
{{
  "answer": "<introduza o resultado com uma frase de contexto, depois destaque os destaques (maior, menor, comparações ou padrões relevantes), como por exemplo: 'Esses são os 10 cargos mais comuns da empresa. O mais frequente é X com Y pessoas, seguido de Z com W. O menos comum entre os listados é A, com B colaboradores.'>",
  "chart_type": "<bar|line|pie|none>",
  "chart_x": "<nome da coluna para eixo X, ou null>",
  "chart_y": "<nome da coluna para eixo Y, ou null>",
  "chart_title": "<título do gráfico, ou null>"
}}

Regras:
- SEMPRE escreva o campo "answer" em português brasileiro.
- NUNCA liste os dados brutos no campo "answer" — interprete-os com linguagem natural.
- Inclua sempre um comentário analítico: destaque o maior, o menor, uma tendência ou comparação relevante nos dados.
- Use chart_type "bar" para comparações, "line" para séries temporais, "pie" para proporções, "none" se tabela basta.
- Formate valores monetários como R$ X.XXX,XX.
- Se os dados estiverem vazios, informe em português que não há dados para o período consultado."""


def _has_valid_key(key: str) -> bool:
    return bool(key) and key not in ("", "COLOQUE_SUA_KEY_AQUI")



def _llm_call(api_key: str, messages: list, max_tokens: int = 512, temperature: float = 0) -> str:
    """Chama LLM via Anthropic SDK ou OpenAI SDK (OpenRouter), retorna texto."""
    if api_key.startswith("sk-or-"):
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://seazone.com.br",
                "X-Title": "Seazone Data Agent",
            },
        )
        resp = client.chat.completions.create(
            model="anthropic/claude-haiku-4-5",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    else:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        return resp.content[0].text.strip()

def _is_numeric(value: str) -> bool:
    try:
        float(value.replace(",", "."))
        return True
    except ValueError:
        return False


def _keyword_parse(answer: str, tables: list[str], labels: list[str]) -> list[str]:
    """Fallback simples para quando não há API key."""
    text = answer.strip().lower()
    all_keywords = {"ambas", "todas", "todos", "ambos", "all", "both", "dois", "duas", "tudo"}
    if any(k in text for k in all_keywords):
        return tables
    for i, table in enumerate(tables):
        if str(i + 1) in text.split():
            return [table]
    for table, label in zip(tables, labels):
        if table.lower() in text or label.lower() in text:
            return [table]
    return tables


class DataAgent:
    def __init__(self, nekt, anthropic_api_key: str):
        self.nekt = nekt
        self.api_key = anthropic_api_key

    def _interpret_with_claude(self, question: str, columns: list, data: list, nekt_explanation: str = "") -> dict:
        prompt = INTERPRET_PROMPT.format(
            question=question,
            nekt_explanation=nekt_explanation or "Não disponível.",
            columns=columns,
            data=data[:20],
        )
        raw = _llm_call(self.api_key, [{"role": "user", "content": prompt}], max_tokens=512)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return json.loads(raw)

    def _interpret_simple(self, columns: list, data: list, nekt_explanation: str = "") -> dict:
        """Fallback sem Claude: formata os dados diretamente."""
        if not data:
            return {"answer": "Não encontrei dados para essa consulta.", "chart_type": "none",
                    "chart_x": None, "chart_y": None, "chart_title": None}

        lines = []
        for row in data[:10]:
            parts = [f"{col}: {val}" for col, val in zip(columns, row)]
            lines.append(" | ".join(parts))
        answer = "\n".join(lines)

        # Heurística de gráfico: 2 colunas com 1ª textual e 2ª numérica → bar
        chart_type = "none"
        chart_x, chart_y = None, None
        if len(columns) == 2:
            try:
                float(data[0][1])
                chart_type = "bar"
                chart_x, chart_y = columns[0], columns[1]
            except (ValueError, IndexError):
                pass

        return {"answer": answer, "chart_type": chart_type,
                "chart_x": chart_x, "chart_y": chart_y, "chart_title": None}

    def parse_clarification_response(self, answer: str, tables: list[str], labels: list[str]) -> list[str]:
        """Usa Claude para interpretar a resposta do usuário e selecionar tabelas.
        Fallback para keyword matching se não houver API key."""
        if not _has_valid_key(self.api_key):
            return _keyword_parse(answer, tables, labels)

        options = "\n".join(f"{i+1}. {label} ({table})" for i, (table, label) in enumerate(zip(tables, labels)))
        prompt = f"""O usuário estava escolhendo entre estas opções de tabelas de dados:

{options}

O usuário respondeu: "{answer}"

Quais tabelas ele quer consultar? Responda SOMENTE com os nomes técnicos das tabelas separados por vírgula.
Se quiser todas, liste todas. Exemplos de resposta: "asaas_transactions" ou "asaas_transactions, postgres_reservations"."""

        raw = _llm_call(self.api_key, [{"role": "user", "content": prompt}], max_tokens=128)
        chosen = [t.strip() for t in raw.split(",") if t.strip() in tables]
        return chosen if chosen else tables

    # Colunas de nome que devem usar ILIKE em vez de = para busca parcial
    _NAME_COLUMNS = {
        'name', 'first_name', 'last_name', 'full_name',
        'employee_name', 'owner_name', 'customer_name',
        'nome', 'sobrenome', 'nome_completo',
    }


    @staticmethod
    def _fix_encoding(sql: str) -> str:
        """Corrige mojibake em literais de string no SQL (UTF-8 interpretado como Latin-1)."""
        import re as _re
        def fix_literal(match):
            val = match.group(1)
            try:
                fixed = val.encode('latin-1').decode('utf-8')
                if fixed != val:
                    return f"'{fixed}'"
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass
            return match.group(0)
        return _re.sub(r"'([^']*)'", fix_literal, sql)

    @staticmethod
    def _fix_name_search(sql: str) -> str:
        import re as _re
        def _replace(match):
            col = match.group(1)
            val = match.group(2)
            col_lower = col.lower()
            if (col_lower in DataAgent._NAME_COLUMNS
                    or col_lower.endswith('_name')
                    or col_lower.endswith('_nome')):
                return f'"{col}" LIKE \'%{val}%\'' 
            return match.group(0)
        return _re.sub(r'"([^"]+)"\s*=\s*\'([^\']+)\'', _replace, sql)

    # Valores de status corretos (PascalCase) — Nekt gera lowercase ou UPPER
    _STATUS_CORRECT = {
        'paid': 'Paid',
        'pending': 'Pending',
        'expired': 'Expired',
        'canceled': 'Canceled',
        'confirmed': 'Confirmed',
        'canceled_to_recreate': 'Canceled_To_Recreate',
        'confirmation_failed': 'Confirmation_Failed',
        'waiting_confirmation': 'Waiting_Confirmation',
        'waiting_cancellation': 'Waiting_Cancellation',
        'active': 'Active',
        'inactive': 'Inactive',
        # dados_churn.fase_do_churn — Nekt gera nomes errados para essas fases
        'churn efetivado': 'Finalizados',
        'churn revertido': 'Revertidos',
    }

    @staticmethod
    def _fix_status_case(sql: str) -> str:
        """Corrige case dos valores de status no SQL gerado pela Nekt."""
        import re as _re
        def _replace(match):
            val = match.group(1)
            corrected = DataAgent._STATUS_CORRECT.get(val.lower(), val)
            return f"'{corrected}'"
        return _re.sub(r"'([^']+)'", _replace, sql)

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        """Transpõe resultados em formato pivô (1 linha, N colunas numéricas) para
        formato longo (categoria | valor), facilitando leitura e geração de gráficos."""
        if len(df) != 1 or len(df.columns) < 2:
            return df

        numeric_count = sum(
            1 for col in df.columns
            if str(df[col].iloc[0]).strip() not in ("", "None", "null")
            and _is_numeric(str(df[col].iloc[0]))
        )

        if numeric_count >= max(1, len(df.columns) // 2):
            transposed = df.T.reset_index()
            transposed.columns = ["categoria", "valor"]
            return transposed

        return df


    _FOLLOWUP_PRONOUNS = {
        'ele', 'ela', 'dele', 'dela', 'desse', 'dessa', 'deles', 'delas',
        'o mesmo', 'a mesma', 'os mesmos', 'as mesmas', 'este', 'esta',
        'esse', 'essa', 'nele', 'nela',
    }

    def _resolve_question(self, question: str, history: list[dict]) -> str:
        """Reescreve a pergunta substituindo pronomes usando o histórico da conversa."""
        q_lower = question.lower()
        if not any(p in q_lower for p in self._FOLLOWUP_PRONOUNS):
            return question
        if not history:
            return question

        # Monta contexto com últimas trocas (máx 4 mensagens)
        ctx_lines = []
        for msg in history[-4:]:
            role = "Usuário" if msg["role"] == "user" else "Assistente"
            ctx_lines.append(f"{role}: {msg['content']}")
        ctx = "\n".join(ctx_lines)

        if not _has_valid_key(self.api_key):
            # Fallback sem Claude: injeta contexto direto na pergunta
            last_user = next(
                (m["content"] for m in reversed(history) if m["role"] == "user"), ""
            )
            return f"{question} (contexto: {last_user})"

        prompt = f"""Reescreva a pergunta do usuário substituindo pronomes e referências implícitas pelo contexto da conversa.
Retorne APENAS a pergunta reescrita, sem explicações.

Histórico:
{ctx}

Pergunta atual: {question}

Pergunta reescrita:"""
        return _llm_call(self.api_key, [{"role": "user", "content": prompt}], max_tokens=128)
        return response.content[0].text.strip()

    def ask(self, question: str, forced_tables: list[str] | None = None, history: list[dict] | None = None) -> dict:

        if history:
            question = self._resolve_question(question, history)

        # Se tabelas não foram forçadas, tenta desambiguar via Nekt
        if forced_tables is None:
            classification = classify_with_tables(question, self.nekt, self.api_key)
            if classification["classificacao"] == "AMBIGUA":
                return {
                    "status": "NEEDS_CLARIFICATION",
                    "pergunta_clarificacao": classification["pergunta_clarificacao"],
                    "tabelas_candidatas": classification["tabelas_candidatas"],
                    "labels_candidatas": classification.get("labels_candidatas", classification["tabelas_candidatas"]),
                    "original_question": question,
                }
            tables = classification["tabelas_candidatas"] or None
            question_hint = classification.get("question_hint")
            column_values = classification.get("column_values")
        else:
            tables = forced_tables
            question_hint = None
            column_values = None

        # Enriquecer pergunta com hint e column_values quando disponíveis.
        # Para multi-tabela: deixar Nekt resolver via busca semântica (sem selected_tables).
        # Para tabela única: passar hint + tabela explicitamente.
        question_for_nekt = question
        nekt_tables = tables

        extra_context = ""
        if question_hint:
            extra_context += question_hint
        if column_values:
            cv_lines = [
                f"  - {col}: valores válidos = {', '.join(repr(v) for v in vals)}"
                for col, vals in column_values.items()
            ]
            extra_context += "\nValores conhecidos das colunas:\n" + "\n".join(cv_lines)

        if extra_context:
            question_for_nekt = f"{question}\n\nContexto técnico: {extra_context}"
            if tables and len(tables) > 1:
                nekt_tables = None

        gen = self.nekt.generate_sql(question_for_nekt, tables=nekt_tables if nekt_tables else None)
        raw_sql = gen.get("sql") or ""
        if not raw_sql:
            raise ValueError("Nekt não conseguiu gerar SQL para essa pergunta.")
        sql = self._fix_name_search(self._fix_status_case(self._fix_encoding(raw_sql)))
        nekt_explanation = gen.get("explanation", "")

        exec_result = self.nekt.execute_sql(sql)
        columns = exec_result.get("columns", [])
        data = exec_result.get("data", [])

        df = self._normalize_df(pd.DataFrame(data, columns=columns)).drop_duplicates()
        columns = list(df.columns)
        data = [[str(v) for v in row] for row in df.values.tolist()]

        if _has_valid_key(self.api_key):
            interpretation = self._interpret_with_claude(question, columns, data, nekt_explanation)
        else:
            interpretation = self._interpret_simple(columns, data, nekt_explanation)

        return {
            "status": "OK",
            "answer": interpretation.get("answer", ""),
            "chart_type": interpretation.get("chart_type", "none"),
            "chart_x": interpretation.get("chart_x"),
            "chart_y": interpretation.get("chart_y"),
            "chart_title": interpretation.get("chart_title"),
            "sql": sql,
            "tables": tables,
            "dataframe": df,
            "columns": columns,
            "data": data,
        }


