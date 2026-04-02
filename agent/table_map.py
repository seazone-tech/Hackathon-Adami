"""E4 — Mapa de temas ambíguos para desambiguação de perguntas."""

TABLE_LABELS = {
    "asaas_transactions": "Transações financeiras (Asaas)",
    "asaas_customers": "Clientes Asaas",
    "google_sheets_rec_ota": "Receita por OTA (Google Sheets)",
    "postgres_reservations": "Reservas (Postgres)",
    "postgres_reservations_scd2": "Reservas histórico (SCD2)",
    "postgres_public_account_owner": "Proprietários (Postgres)",
    "sapron_public_reservation_listing": "Listagem de reservas (Sapron)",
    "leads_abertos": "Leads abertos",
    "deals_comissionamento": "Deals de comissionamento",
    "postgres_website_public_reservations": "Reservas do website",
    "checking_accounts_balance": "Saldo de contas correntes",
    "google_sheets_metas_q2_2025": "Metas Q2 2025",
    "pipedrive_v2_deal_flow": "Deal flow (Pipedrive)",
    "convenia_employees": "Funcionários (Convenia)",
    "kobana_kobana_billets": "Boletos (Kobana)",
    "people_colaboradores": "Colaboradores — nome completo e salário (Silver)",
    "convenia_employees_transformada": "Funcionários ativos (contagem e dados atuais)",
    "dados_churn": "Churn de proprietários — histórico de saídas (Silver)",
    "pipefy_szs_all_cards_303781436_colunas_expandidas": "Imóveis por franqueado — anfitrião responsável por imóvel",
}


def get_table_label(table_name: str) -> str:
    """Retorna label amigável para uma tabela. Fallback: humaniza o nome."""
    if table_name in TABLE_LABELS:
        return TABLE_LABELS[table_name]
    return table_name.replace("_", " ").title()


TABLE_OVERLAP_MAP = {
    "faturamento": {
        "keywords": ["fatur", "receita", "quanto ganhamos", "rendimento", "revenue"],
        "candidates": [
            {
                "table": "postgres_reservations",
                "label": "Faturamento por reservas pagas (dados até out/2025)",
            },
            {
                "table": "asaas_transactions",
                "label": "Faturamento financeiro — cobranças Asaas (dados até jan/2025)",
            },
            {
                "table": "google_sheets_rec_ota",
                "label": "Receita por OTA (Airbnb, Booking, etc.)",
            },
        ],
    },
    "receita_ota": {
        "keywords": ["receita ota", "canal", "airbnb", "booking", "plataforma"],
        "candidates": [
            {
                "table": "google_sheets_rec_ota",
                "label": "Receita por OTA (planilha)",
            },
            {
                "table": "sapron_public_reservation_listing",
                "label": "Listagem de reservas por OTA (Sapron)",
            },
        ],
    },
    "proprietarios": {
        "keywords": ["proprietário", "proprietario", "dono", "owner", "churn proprietário"],
        "candidates": [
            {
                "table": "postgres_public_account_owner",
                "label": "Cadastro de proprietários",
            },
            {
                "table": "postgres_reservations",
                "label": "Reservas dos imóveis dos proprietários",
            },
        ],
    },
    "imoveis": {
        "keywords": ["imóvel", "imovel", "imóveis", "imoveis", "propriedade", "unidade"],
        "candidates": [
            {
                "table": "postgres_reservations",
                "label": "Reservas por imóvel",
            },
            {
                "table": "sapron_public_reservation_listing",
                "label": "Listagem/performance por imóvel (Sapron)",
            },
        ],
    },
}


# Tabelas preferenciais para temas sem ambiguidade — hint passada ao Nekt
# Entradas com "tables" (lista) têm prioridade e são verificadas antes do TABLE_OVERLAP_MAP.
TABLE_PREFERRED_MAP = {
    "churn_revertido": {
        "keywords": ["revertido", "revertidos", "revers", "recuperado", "recuperados"],
        "tables": ["dados_churn"],
        "question_hint": (
            "Use APENAS a tabela nekt_silver.dados_churn. "
            "Filtre pela coluna fase_do_churn = 'Revertidos' para contar churns revertidos/recuperados. "
            "Não faça JOIN com nenhuma outra tabela."
        ),
        "column_values": {
            "fase_do_churn": ["Finalizados", "Revertidos"],
        },
    },
    "churn_proprietario": {
        "keywords": ["churn", "distrato"],
        "tables": [
            "dados_churn",
            "pipefy_szs_all_cards_303781436_colunas_expandidas",
        ],
        "question_hint": (
            "Use a tabela nekt_silver.dados_churn para os registros de churn. "
            "A coluna fase_do_churn indica o status: 'Finalizados' = churn efetivado, 'Revertidos' = churn revertido. "
            "Faça JOIN com nekt_service.pipefy_szs_all_cards_303781436_colunas_expandidas "
            "usando dados_churn.codigo_do_imovel = pipefy_szs_all_cards_303781436_colunas_expandidas.title "
            "apenas quando precisar do nome do franqueado (anfitriao_responsavel)."
        ),
        "column_values": {
            "fase_do_churn": ["Finalizados", "Revertidos"],
        },
    },
    "funcionarios_atributo": {
        "keywords": [
            "salário", "salario", "cargo",
            "admissão", "admissao", "demissão", "demissao",
        ],
        "table": "people_colaboradores",
    },
    "funcionarios_contagem": {
        "keywords": [
            "funcionário", "funcionario", "funcionários", "funcionarios",
            "colaborador", "colaboradores", "headcount", "contratado",
            "trabalhador", "empregado",
        ],
        "table": "convenia_employees_transformada",
    },
}


_TABLE_ROUTES = [
    (["fatur", "receita", "transaç", "pagamento", "financ", "revenue", "cobranç"],
     ["asaas_transactions"]),
    (["reserva", "check-in", "checkout", "imovel", "imóvel", "hospedagem", "diária", "noite"],
     ["postgres_reservations"]),
    (["proprietário", "proprietario", "dono", "owner", "churn", "franqueado"],
     ["postgres_public_account_owner", "postgres_reservations"]),
    (["lead", "comercial", "pipeline", "venda"],
     ["leads_abertos"]),
    (["deal", "comissão", "comissao"],
     ["deals_comissionamento"]),
    (["ota", "airbnb", "booking", "canal"],
     ["sapron_public_reservation_listing", "google_sheets_rec_ota"]),
]

_DEFAULT_TABLES = ["asaas_transactions", "postgres_reservations"]


def route_tables(question: str) -> list[str]:
    """Retorna lista de tabelas relevantes baseado em keywords na pergunta."""
    q = question.lower()
    matched = []
    for keywords, tables in _TABLE_ROUTES:
        if any(kw in q for kw in keywords):
            for t in tables:
                if t not in matched:
                    matched.append(t)
    return matched if matched else _DEFAULT_TABLES


def build_schema_context() -> str:
    """Serializa TABLE_OVERLAP_MAP como texto para uso em prompts."""
    lines = []
    for theme, info in TABLE_OVERLAP_MAP.items():
        lines.append(f"Tema: {theme}")
        lines.append(f"  Keywords: {', '.join(info['keywords'])}")
        lines.append("  Tabelas candidatas:")
        for c in info["candidates"]:
            lines.append(f"    - {c['table']}: {c['label']}")
    return "\n".join(lines)
