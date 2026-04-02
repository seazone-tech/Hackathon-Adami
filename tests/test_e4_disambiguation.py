"""Tests for E4 — Disambiguation: table_map + disambiguator + agent integration."""

from unittest.mock import MagicMock


def test_table_overlap_map_has_required_themes():
    from agent.table_map import TABLE_OVERLAP_MAP

    assert "faturamento" in TABLE_OVERLAP_MAP
    assert "proprietarios" in TABLE_OVERLAP_MAP
    assert "imoveis" in TABLE_OVERLAP_MAP
    assert "receita_ota" in TABLE_OVERLAP_MAP


def test_build_schema_context_returns_nonempty_string():
    from agent.table_map import build_schema_context

    ctx = build_schema_context()
    assert isinstance(ctx, str)
    assert len(ctx) > 0
    assert "faturamento" in ctx
    assert "asaas_transactions" in ctx


def test_table_labels_has_known_tables():
    from agent.table_map import TABLE_LABELS

    assert "asaas_transactions" in TABLE_LABELS
    assert "postgres_reservations" in TABLE_LABELS
    assert "google_sheets_rec_ota" in TABLE_LABELS


def test_get_table_label_known():
    from agent.table_map import get_table_label

    assert get_table_label("asaas_transactions") == "Transações financeiras (Asaas)"


def test_get_table_label_unknown_humanizes():
    from agent.table_map import get_table_label

    assert get_table_label("some_weird_table") == "Some Weird Table"


def test_classify_with_tables_single_table_is_clara():
    """Pergunta sem tema ambíguo → classificação CLARA, Nekt decide a tabela."""
    from agent.disambiguator import classify_with_tables

    nekt = MagicMock()

    result = classify_with_tables("Qual o saldo das contas correntes?", nekt, api_key=None)
    assert result["classificacao"] == "CLARA"
    assert result["tabelas_candidatas"] == []


def test_classify_with_tables_multiple_is_ambigua():
    """Pergunta que bate em tema ambíguo → AMBIGUA com múltiplas opções."""
    from agent.disambiguator import classify_with_tables

    nekt = MagicMock()

    result = classify_with_tables("Quanto faturamos?", nekt, api_key=None)
    assert result["classificacao"] == "AMBIGUA"
    assert len(result["tabelas_candidatas"]) >= 2
    assert len(result["labels_candidatas"]) >= 2
    assert result["pergunta_clarificacao"] is not None


def test_classify_with_tables_nekt_error_falls_back():
    """Se Nekt falha, cai no fallback keyword-based."""
    from agent.disambiguator import classify_with_tables

    nekt = MagicMock()
    nekt.get_relevant_tables.side_effect = RuntimeError("connection error")

    result = classify_with_tables("Quanto faturamos?", nekt, api_key=None)
    assert result["classificacao"] in ("CLARA", "AMBIGUA")
    assert "tabelas_candidatas" in result


def test_classify_with_tables_nekt_empty_falls_back():
    """Se Nekt retorna lista vazia, cai no fallback."""
    from agent.disambiguator import classify_with_tables

    nekt = MagicMock()
    nekt.get_relevant_tables.return_value = []

    result = classify_with_tables("Quanto faturamos?", nekt, api_key=None)
    assert result["classificacao"] in ("CLARA", "AMBIGUA")


def test_classify_with_tables_limits_to_max():
    """Limita a MAX_TABLE_OPTIONS tabelas."""
    from agent.disambiguator import classify_with_tables, MAX_TABLE_OPTIONS

    nekt = MagicMock()
    nekt.get_relevant_tables.return_value = [f"table_{i}" for i in range(10)]

    result = classify_with_tables("pergunta genérica", nekt, api_key=None)
    assert len(result["tabelas_candidatas"]) <= MAX_TABLE_OPTIONS


def test_agent_ask_returns_needs_clarification_for_ambiguous():
    """agent.ask() com Nekt retornando múltiplas tabelas → NEEDS_CLARIFICATION."""
    from agent.agent import DataAgent

    nekt = MagicMock()
    nekt.get_relevant_tables.return_value = ["asaas_transactions", "google_sheets_rec_ota"]

    agent = DataAgent(nekt=nekt, anthropic_api_key=None)
    result = agent.ask("Quanto faturamos?")

    assert result["status"] == "NEEDS_CLARIFICATION"
    assert "pergunta_clarificacao" in result
    assert "tabelas_candidatas" in result
    assert "labels_candidatas" in result


def test_agent_ask_with_forced_tables_skips_disambiguation():
    """agent.ask() com forced_tables deve pular desambiguação e retornar dados."""
    from agent.agent import DataAgent

    nekt = MagicMock()
    nekt.generate_sql.return_value = {"sql": "SELECT 1"}
    nekt.execute_sql.return_value = {"columns": ["val"], "data": [[1]]}

    agent = DataAgent(nekt=nekt, anthropic_api_key=None)
    result = agent.ask("Quanto faturamos?", forced_tables=["asaas_transactions"])

    assert result.get("status") == "OK"
    assert "sql" in result
    nekt.get_relevant_tables.assert_not_called()


def test_agent_ask_clara_goes_straight_to_sql():
    """Nekt retorna 1 tabela → vai direto para SQL sem pedir clarificação."""
    from agent.agent import DataAgent

    nekt = MagicMock()
    nekt.get_relevant_tables.return_value = ["postgres_reservations"]
    nekt.generate_sql.return_value = {"sql": "SELECT count(*) FROM reservations"}
    nekt.execute_sql.return_value = {"columns": ["count"], "data": [[42]]}

    agent = DataAgent(nekt=nekt, anthropic_api_key=None)
    result = agent.ask("Quantas reservas temos?")

    assert result["status"] == "OK"
    assert result["sql"] == "SELECT count(*) FROM reservations"


def test_classify_with_tables_employee_returns_preferred_table():
    """Perguntas de funcionário/salário devem retornar CLARA com tabela preferencial."""
    from agent.disambiguator import classify_with_tables

    nekt = MagicMock()

    result = classify_with_tables("qual o salário de Rivaldo Felinto?", nekt, api_key=None)
    assert result["classificacao"] == "CLARA"
    assert "people_colaboradores" in result["tabelas_candidatas"]


def test_classify_headcount_does_not_route_to_people_colaboradores():
    """Queries de contagem de funcionários NÃO devem forçar people_colaboradores."""
    from agent.disambiguator import classify_with_tables

    nekt = MagicMock()

    result = classify_with_tables("quantos funcionários ativos na empresa?", nekt, api_key=None)
    assert "people_colaboradores" not in result["tabelas_candidatas"]


def test_classify_headcount_routes_to_employees_transformada():
    """Queries de contagem de funcionários devem usar convenia_employees_transformada."""
    from agent.disambiguator import classify_with_tables

    nekt = MagicMock()

    result = classify_with_tables("quantos funcionários ativos na empresa?", nekt, api_key=None)
    assert result["classificacao"] == "CLARA"
    assert "convenia_employees_transformada" in result["tabelas_candidatas"]


def test_table_labels_has_dados_churn():
    """dados_churn deve ter label amigável no TABLE_LABELS."""
    from agent.table_map import TABLE_LABELS

    assert "dados_churn" in TABLE_LABELS


def test_table_labels_has_imovel_franqueado_table():
    from agent.table_map import TABLE_LABELS

    assert "pipefy_szs_all_cards_303781436_colunas_expandidas" in TABLE_LABELS


def test_classify_churn_is_clara():
    """'churn de proprietários' deve ser CLARA, não pedir clarificação."""
    from agent.disambiguator import classify_with_tables

    nekt = MagicMock()
    result = classify_with_tables("Qual franqueado tem maior churn de proprietários?", nekt, api_key=None)
    assert result["classificacao"] == "CLARA"
    assert result["pergunta_clarificacao"] is None


def test_classify_churn_routes_to_dados_churn():
    from agent.disambiguator import classify_with_tables

    nekt = MagicMock()
    result = classify_with_tables("Qual franqueado tem maior churn de proprietários?", nekt, api_key=None)
    assert "dados_churn" in result["tabelas_candidatas"]


def test_classify_churn_routes_to_imovel_table():
    """Deve incluir tabela de imóveis para o JOIN com franqueado."""
    from agent.disambiguator import classify_with_tables

    nekt = MagicMock()
    result = classify_with_tables("Qual franqueado tem maior churn de proprietários?", nekt, api_key=None)
    assert "pipefy_szs_all_cards_303781436_colunas_expandidas" in result["tabelas_candidatas"]


def test_classify_churn_takes_priority_over_proprietarios_ambiguity():
    """'churn' deve ter prioridade sobre o tema ambíguo 'proprietarios'."""
    from agent.disambiguator import classify_with_tables

    nekt = MagicMock()
    result = classify_with_tables("quantos churns de proprietários este mês?", nekt, api_key=None)
    assert result["classificacao"] == "CLARA"


def test_status_correct_has_churn_efetivado():
    from agent.agent import DataAgent

    assert DataAgent._STATUS_CORRECT.get("churn efetivado") == "Finalizados"


def test_status_correct_has_churn_revertido():
    from agent.agent import DataAgent

    assert DataAgent._STATUS_CORRECT.get("churn revertido") == "Revertidos"


def test_fix_status_case_corrects_churn_efetivado():
    from agent.agent import DataAgent

    sql = """SELECT * FROM dados_churn WHERE "fase_do_churn" = 'Churn Efetivado'"""
    fixed = DataAgent._fix_status_case(sql)
    assert "'Finalizados'" in fixed
    assert "'Churn Efetivado'" not in fixed
