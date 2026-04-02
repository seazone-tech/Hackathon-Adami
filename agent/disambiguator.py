"""E4 — Classificador de perguntas ambíguas."""

import json
import anthropic
from agent.table_map import TABLE_LABELS, TABLE_OVERLAP_MAP, TABLE_PREFERRED_MAP, build_schema_context, route_tables, get_table_label


CLASSIFY_PROMPT = """Você é um classificador de perguntas de negócio da Seazone.

Dado o contexto de tabelas disponíveis e seus temas ambíguos:
{schema_context}

Classifique a pergunta do usuário:
Pergunta: "{question}"

Responda SOMENTE com JSON válido, sem markdown:
{{
  "classificacao": "CLARA" ou "AMBIGUA",
  "tabela_principal": "<tabela mais provável, ou null se ambígua>",
  "tabelas_candidatas": ["<tabela1>", "<tabela2>"],
  "pergunta_clarificacao": "<pergunta para o usuário escolher, ou null se CLARA>",
  "explicacao_abordagem": "<motivo da classificação>"
}}

Regras:
- CLARA: a pergunta aponta claramente para um único domínio/tabela
- AMBIGUA: a pergunta pode se referir a múltiplos domínios (ex: "receita" pode ser Asaas ou OTA)
- Se AMBIGUA, tabelas_candidatas deve ter 2+ opções e pergunta_clarificacao deve ser direta
- Se CLARA, tabelas_candidatas pode listar a tabela principal apenas"""



def _llm_call(api_key: str, messages: list, max_tokens: int = 256, temperature: float = 0) -> str:
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

def _fallback_classify(question: str) -> dict:
    """Classificação sem API key: usa route_tables como heurística."""
    tables = route_tables(question)
    q = question.lower()

    # Detecta se a pergunta bate em múltiplos temas do TABLE_OVERLAP_MAP
    matched_themes = []
    for theme, info in TABLE_OVERLAP_MAP.items():
        if any(kw in q for kw in info["keywords"]):
            matched_themes.append(theme)

    if len(matched_themes) >= 1:
        theme = matched_themes[0]
        candidates = [c["table"] for c in TABLE_OVERLAP_MAP[theme]["candidates"]]
        labels = [c["label"] for c in TABLE_OVERLAP_MAP[theme]["candidates"]]
        return {
            "classificacao": "AMBIGUA",
            "tabela_principal": None,
            "tabelas_candidatas": candidates,
            "labels_candidatas": labels,
            "pergunta_clarificacao": "Sua pergunta pode se referir a duas fontes diferentes. Escolha:",
            "explicacao_abordagem": f"Pergunta bate no tema ambíguo '{theme}'",
        }

    return {
        "classificacao": "CLARA",
        "tabela_principal": tables[0] if tables else None,
        "tabelas_candidatas": tables,
        "pergunta_clarificacao": None,
        "explicacao_abordagem": "Pergunta aponta claramente para um domínio",
    }


def classify_question(question: str, api_key: str | None) -> dict:
    """
    Classifica uma pergunta como CLARA ou AMBIGUA.

    Retorna dict com:
      classificacao, tabela_principal, tabelas_candidatas,
      pergunta_clarificacao, explicacao_abordagem
    """
    if not api_key or api_key in ("", "COLOQUE_SUA_KEY_AQUI"):
        return _fallback_classify(question)

    prompt = CLASSIFY_PROMPT.format(
        schema_context=build_schema_context(),
        question=question,
    )
    raw = _llm_call(api_key, [{"role": "user", "content": prompt}], max_tokens=256)
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


MAX_TABLE_OPTIONS = 5


def classify_with_tables(question: str, nekt, api_key: str | None) -> dict:
    """
    Classificação híbrida:
    1. Checa se a pergunta bate num tema ambíguo (TABLE_OVERLAP_MAP)
       → Se sim, pede clarificação ao usuário
    2. Senão → CLARA sem tabelas forçadas (Nekt generate_sql escolhe sozinha)
    """
    q = question.lower()

    # Passo 1: checar tabela preferencial (mais específico — tem prioridade sobre ambiguidade)
    for theme, info in TABLE_PREFERRED_MAP.items():
        if any(kw in q for kw in info["keywords"]):
            tables = info.get("tables") or [info["table"]]
            return {
                "classificacao": "CLARA",
                "tabela_principal": tables[0],
                "tabelas_candidatas": tables,
                "labels_candidatas": [TABLE_LABELS.get(t, t) for t in tables],
                "pergunta_clarificacao": None,
                "explicacao_abordagem": f"Tabela preferencial para {theme}",
                "question_hint": info.get("question_hint"),
                "column_values": info.get("column_values"),
            }

    # Passo 2: verificar se bate em tema ambíguo
    for theme, info in TABLE_OVERLAP_MAP.items():
        if any(kw in q for kw in info["keywords"]):
            candidates = [c["table"] for c in info["candidates"]]
            labels = [c["label"] for c in info["candidates"]]
            return {
                "classificacao": "AMBIGUA",
                "tabela_principal": None,
                "tabelas_candidatas": candidates,
                "labels_candidatas": labels,
                "pergunta_clarificacao": "Sua pergunta pode se referir a fontes diferentes. Escolha:",
                "explicacao_abordagem": f"Tema ambiguo: {theme}",
            }

    # Passo 3: sem hint → Nekt escolhe sozinha
    return {
        "classificacao": "CLARA",
        "tabela_principal": None,
        "tabelas_candidatas": [],
        "labels_candidatas": [],
        "pergunta_clarificacao": None,
        "explicacao_abordagem": "Sem ambiguidade — Nekt escolhe a tabela",
    }
