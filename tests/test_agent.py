"""Tests for the DataAgent — orchestrates Nekt + Claude to answer business questions."""
from unittest.mock import MagicMock, patch


def _make_nekt(sql="SELECT 1", columns=None, data=None):
    nekt = MagicMock()
    nekt.generate_sql.return_value = {"sql": sql, "is_valid": True, "explanation": "ok"}
    nekt.execute_sql.return_value = {
        "status": "succeeded",
        "columns": columns or ["month", "total"],
        "data": data or [["2025-02", "100000"], ["2025-03", "120000"]],
        "row_count": 2,
    }
    return nekt


def test_ask_returns_answer_and_sql():
    from agent.agent import DataAgent

    nekt = _make_nekt()
    agent = DataAgent(nekt=nekt, anthropic_api_key="fake-key")

    mock_claude = MagicMock()
    mock_claude.messages.create.return_value = MagicMock(
        content=[MagicMock(text='{"answer": "Fevereiro: R$100k, Março: R$120k.", "chart_type": "bar", "chart_x": "month", "chart_y": "total"}')]
    )

    with patch("agent.agent.classify_with_tables") as mock_classify, \
         patch("agent.agent.anthropic.Anthropic", return_value=mock_claude):
        mock_classify.return_value = {
            "classificacao": "CLARA",
            "tabela_principal": "asaas_transactions",
            "tabelas_candidatas": ["asaas_transactions"],
            "pergunta_clarificacao": None,
            "explicacao_abordagem": "ok",
        }
        result = agent.ask("Quanto faturamos em fevereiro vs março?")

    assert "answer" in result
    assert "sql" in result
    assert "dataframe" in result
    assert result["answer"] == "Fevereiro: R$100k, Março: R$120k."
    assert result["sql"] == "SELECT 1"


def test_ask_propagates_nekt_error():
    from agent.agent import DataAgent
    import pytest

    nekt = MagicMock()
    nekt.generate_sql.return_value = {"sql": "SELECT 1", "is_valid": True}
    nekt.execute_sql.side_effect = RuntimeError("Nekt error: bad sql")

    agent = DataAgent(nekt=nekt, anthropic_api_key="fake")
    with pytest.raises(RuntimeError, match="Nekt error"):
        agent.ask("qualquer pergunta", forced_tables=["asaas_transactions"])


def test_ask_uses_simple_fallback_when_no_api_key():
    from agent.agent import DataAgent

    nekt = _make_nekt()
    agent = DataAgent(nekt=nekt, anthropic_api_key="")

    # forced_tables pula desambiguação e vai direto para consulta
    result = agent.ask("Quanto faturamos?", forced_tables=["asaas_transactions"])

    assert "answer" in result
    assert result["answer"] != ""
    assert "sql" in result
    assert "dataframe" in result


def test_ask_with_forced_tables_calls_generate_sql_with_those_tables():
    from agent.agent import DataAgent

    nekt = MagicMock()
    nekt.generate_sql.return_value = {"sql": "SELECT 1", "is_valid": True}
    nekt.execute_sql.return_value = {
        "status": "succeeded", "columns": ["v"], "data": [["1"]], "row_count": 1
    }

    agent = DataAgent(nekt=nekt, anthropic_api_key="")
    agent.ask("Quanto faturamos?", forced_tables=["asaas_transactions"])

    nekt.generate_sql.assert_called_once_with("Quanto faturamos?", tables=["asaas_transactions"])


def test_route_tables_for_revenue_question():
    from agent.agent import route_tables

    tables = route_tables("Quanto faturamos em março?")
    assert "asaas_transactions" in tables


def test_route_tables_for_reservations_question():
    from agent.agent import route_tables

    tables = route_tables("Quais imóveis têm mais dias sem reserva?")
    assert "postgres_reservations" in tables


def test_route_tables_for_owner_question():
    from agent.agent import route_tables

    tables = route_tables("Qual franqueado tem maior churn de proprietários?")
    assert "postgres_public_account_owner" in tables


def test_fix_name_search_replaces_exact_with_ilike():
    from agent.agent import DataAgent

    sql = """SELECT * FROM employees WHERE "name" = 'Felipe' AND "last_name" = 'Adami'"""
    result = DataAgent._fix_name_search(sql)
    assert '"name" LIKE \'%Felipe%\'' in result
    assert '"last_name" LIKE \'%Adami%\'' in result


def test_fix_name_search_does_not_touch_non_name_columns():
    from agent.agent import DataAgent

    sql = """SELECT * FROM employees WHERE "status" = 'active' AND "name" = 'Felipe'"""
    result = DataAgent._fix_name_search(sql)
    assert '"status" = \'active\'' in result
    assert '"name" LIKE \'%Felipe%\'' in result


def test_fix_name_search_applied_in_ask():
    from agent.agent import DataAgent

    raw_sql = """SELECT * FROM employees WHERE "name" = 'Felipe' AND "last_name" = 'Adami'"""
    nekt = _make_nekt(sql=raw_sql, columns=["name", "role"], data=[["Felipe Adami", "Engineer"]])
    agent = DataAgent(nekt=nekt, anthropic_api_key="")

    result = agent.ask("qual o cargo do funcionario Felipe Adami", forced_tables=["convenia_employees"])

    assert "LIKE" in result["sql"]
    assert "= 'Felipe'" not in result["sql"]


def test_fix_name_search_uses_like_not_ilike():
    from agent.agent import DataAgent

    sql = """SELECT * FROM employees WHERE "name" = 'Felipe'"""
    result = DataAgent._fix_name_search(sql)
    assert "LIKE '%Felipe%'" in result
    assert "ILIKE" not in result


def test_ask_resolves_pronoun_with_history():
    """Pergunta de follow-up com pronome deve ser reescrita com contexto do histórico."""
    from agent.agent import DataAgent
    from unittest.mock import patch, MagicMock

    nekt = _make_nekt(columns=["name", "salary"], data=[["Felipe Adami", "10000"]])
    agent = DataAgent(nekt=nekt, anthropic_api_key="")

    history = [
        {"role": "user", "content": "qual o cargo do funcionário Felipe Adami"},
        {"role": "assistant", "content": "Felipe Adami é Engenheiro de Software."},
    ]

    result = agent.ask("qual o salário dele?", forced_tables=["convenia_employees"], history=history)

    # A pergunta enviada ao Nekt deve conter "Felipe Adami", não apenas "dele"
    call_args = nekt.generate_sql.call_args
    resolved_question = call_args[0][0]
    assert "Felipe Adami" in resolved_question


def test_ask_deduplicates_identical_rows():
    """Linhas duplicadas retornadas pelo Nekt devem ser colapsadas em uma."""
    from agent.agent import DataAgent

    nekt = _make_nekt(
        columns=["name", "salary"],
        data=[["Flavio Nineland", "4000.0"]] * 10,
    )
    agent = DataAgent(nekt=nekt, anthropic_api_key="")

    result = agent.ask("qual o salario de Flavio Nineland", forced_tables=["convenia_employees"])

    assert len(result["dataframe"]) == 1


def test_fix_status_case_corrects_active():
    from agent.agent import DataAgent

    sql = """SELECT COUNT("id") FROM employees WHERE "status" = 'active'"""
    result = DataAgent._fix_status_case(sql)
    assert "'Active'" in result
    assert "'active'" not in result
