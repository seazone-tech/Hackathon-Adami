import os
import base64

import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

from agent.nekt_client import NektClient
from agent.agent import DataAgent

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Seazone Data Agent",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: #0a0e1a; }
  .block-container { padding-top: 3.5rem !important; max-width: 860px !important; }

  .sz-header { display: flex; align-items: center; gap: 16px; margin-bottom: 6px; }
  .sz-badge {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8);
    color: white; font-weight: 700; font-size: 12px; letter-spacing: 2px;
    padding: 6px 14px; border-radius: 8px; text-transform: uppercase;
  }
  .sz-title { font-size: 26px; font-weight: 800; color: #f1f5f9; margin: 0; }
  .sz-subtitle { font-size: 14px; color: #64748b; margin: 4px 0 0; }
  .sz-divider { border: none; border-top: 1px solid #1e293b; margin: 14px 0 20px; }
  .sz-section {
    font-size: 11px; font-weight: 600; color: #475569;
    text-transform: uppercase; letter-spacing: 1.4px; margin-bottom: 10px;
  }

  div[data-testid="column"] .stButton > button {
    width: 100%; background: #0f1628; border: 1px solid #1e3a5f;
    border-radius: 12px; color: #94a3b8; font-size: 13px; font-weight: 500;
    padding: 14px 16px; text-align: left; line-height: 1.5;
    transition: all 0.18s ease; min-height: 72px; white-space: normal;
  }
  div[data-testid="column"] .stButton > button:hover {
    background: #0f1d32; border-color: #3b82f6; color: #e2e8f0;
    transform: translateY(-1px); box-shadow: 0 4px 20px rgba(59,130,246,0.15);
  }

  .stChatInput textarea {
    background: #0f1628 !important; border: 1px solid #1e3a5f !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
    font-size: 13px !important; min-height: 44px !important; padding: 10px 14px !important;
  }
  .stChatInput textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
  }
  div[data-testid="stChatInput"] { padding-left: 160px !important; }
  div[data-testid="stChatInput"] > div { min-height: 48px !important; }

  /* Mensagem do usuário — direita */
  .sz-user-msg {
    display: flex;
    justify-content: flex-end;
    align-items: flex-start;
    gap: 10px;
    margin: 6px 0 14px;
  }
  .sz-user-bubble {
    background: #1e3a5f;
    border: 1px solid #2563eb;
    border-radius: 14px 2px 14px 14px;
    color: #e2e8f0;
    padding: 12px 16px;
    max-width: 72%;
    font-size: 14px;
    line-height: 1.5;
  }
  .sz-user-avatar {
    width: 36px; height: 36px; min-width: 36px;
    background: #e55c4a;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
  }

  /* Mensagem do assistente — esquerda */
  [data-testid="stChatMessageContent"] {
    background: #0f1628 !important;
    border: 1px solid #1e293b !important;
    border-radius: 2px 14px 14px 14px !important;
    color: #e2e8f0 !important;
    margin-right: 15%;
  }

  .stDataFrame { border-radius: 10px; overflow: hidden; border: 1px solid #1e293b; }

  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #0a0e1a; }
  ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #3b82f6; }
</style>
""", unsafe_allow_html=True)

# ── Tópicos sensíveis ─────────────────────────────────────────────────────────
_SENSITIVE_KEYWORDS = [
    # salário / remuneração
    "salário", "salario", "salários", "salarios",
    "remuneração", "remuneracao", "holerite", "contracheque",
    "folha de pagamento", "folha salarial",
    # desligamentos
    "desligamento", "desligamentos", "demissão", "demissao",
    "demissões", "demissoes", "demitido", "demitidos",
    "rescisão", "rescisao", "rescisões", "rescisoes",
    # processos judiciais
    "processo judicial", "processos judiciais",
    "ação judicial", "acao judicial",
    "ações judiciais", "acoes judiciais",
    "litígio", "litigio", "litígios", "litigios",
    "vara do trabalho", "tribunal",
]

_SENSITIVE_MSG = "Informação sensível. Por favor, verifique a informação com o time responsável."


def _is_sensitive(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _SENSITIVE_KEYWORDS)


# ── Sugestões ─────────────────────────────────────────────────────────────────
SUGGESTIONS = [
    ("👥", "Headcount",   "Quantos funcionários ativos temos?"),
    ("📉", "Churn",       "Qual franqueado tem maior churn de proprietários?"),
    ("💼", "Cargos",      "Quais os 10 cargos mais comuns na empresa?"),
    ("↩️", "Retenção",    "Quantos churns foram revertidos?"),
    ("📡", "OTAs",        "Qual OTA gerou mais receita?"),
    ("💰", "Faturamento", "Quanto faturamos em março vs fevereiro?"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_chart(df: pd.DataFrame, chart_type: str, chart_x, chart_y, title):
    if chart_type == "none" or not chart_x or not chart_y:
        return None
    if chart_x not in df.columns or chart_y not in df.columns:
        return None

    df = df.copy()
    df[chart_y] = pd.to_numeric(df[chart_y], errors="coerce")

    colors = ["#3b82f6", "#60a5fa", "#93c5fd", "#1d4ed8", "#2563eb"]
    kwargs = dict(x=chart_x, y=chart_y, title=title or "", template="plotly_dark",
                  color_discrete_sequence=colors)

    if chart_type == "bar":
        fig = px.bar(df, **kwargs)
    elif chart_type == "line":
        fig = px.line(df, **kwargs)
    elif chart_type == "pie":
        fig = px.pie(df, names=chart_x, values=chart_y, title=title or "",
                     template="plotly_dark", color_discrete_sequence=colors)
    else:
        return None

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0c1020",
        font_color="#94a3b8",
        title_font_color="#e2e8f0",
        title_font_size=15,
        margin=dict(l=16, r=16, t=48, b=16),
        xaxis=dict(gridcolor="#1e293b", linecolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b", linecolor="#1e293b"),
    )
    return fig


def _build_clarification_message(base_msg: str, tables: list, labels: list) -> str:
    lines = [base_msg, ""]
    for i, label in enumerate(labels):
        lines.append(f"{i + 1}. {label}")
    lines.append("")
    lines.append("Qual você quer consultar? Digite o número, o nome ou **'ambas'** para as duas.")
    return "\n".join(lines)


@st.cache_resource
def get_agent():
    nekt = NektClient(
        url=os.getenv("NEKT_MCP_URL", "https://nekt-mcp.seazone.com.br/mcp"),
        token=os.getenv("NEKT_TOKEN", ""),
    )
    return DataAgent(nekt=nekt, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""))


def _render_result(result: dict):
    answer = result["answer"]
    fig = build_chart(
        result["dataframe"], result["chart_type"],
        result["chart_x"], result["chart_y"], result["chart_title"],
    )
    df = result["dataframe"] if not result["dataframe"].empty else None

    st.markdown(answer)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    if df is not None:
        st.dataframe(df, use_container_width=True, hide_index=True)
    with st.expander("🔍 SQL gerado"):
        if result.get("tables"):
            st.caption("Tabelas: " + " · ".join(f"`{t}`" for t in result["tables"]))
        st.code(result["sql"], language="sql")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chart": fig,
        "dataframe": df,
        "sql": result["sql"],
        "tables": result.get("tables"),
    })


def _handle_clarification(answer: str):
    ctx = st.session_state.awaiting_clarification
    agent = get_agent()
    forced = agent.parse_clarification_response(
        answer, ctx["tabelas_candidatas"], ctx.get("labels_candidatas", ctx["tabelas_candidatas"])
    )
    st.session_state.awaiting_clarification = None
    with st.chat_message("assistant"):
        with st.spinner("Consultando data lake..."):
            try:
                result = agent.ask(ctx["original_question"], forced_tables=forced)
                _render_result(result)
            except Exception as e:
                st.error(f"Não consegui responder: {e}")
    st.rerun()


def _render_user_msg(text: str):
    st.markdown(
        f'<div class="sz-user-msg">'
        f'<div class="sz-user-bubble">{text}</div>'
        f'<div class="sz-user-avatar">👤</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _process_question(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    _render_user_msg(question)

    if _is_sensitive(question):
        with st.chat_message("assistant"):
            st.warning(_SENSITIVE_MSG)
        st.session_state.messages.append({"role": "assistant", "content": _SENSITIVE_MSG})
        return

    with st.chat_message("assistant"):
        with st.spinner("Consultando data lake..."):
            try:
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[-7:-1]
                    if m.get("content")
                ]
                result = get_agent().ask(question, history=history if len(history) > 1 else None)

                if result["status"] == "NEEDS_CLARIFICATION":
                    labels = result.get("labels_candidatas", result["tabelas_candidatas"])
                    clarif_msg = _build_clarification_message(
                        result["pergunta_clarificacao"],
                        result["tabelas_candidatas"],
                        labels,
                    )
                    st.session_state.awaiting_clarification = {
                        "original_question": result["original_question"],
                        "tabelas_candidatas": result["tabelas_candidatas"],
                        "labels_candidatas": labels,
                    }
                    st.markdown(clarif_msg)
                    st.session_state.messages.append({"role": "assistant", "content": clarif_msg})
                    st.rerun()
                else:
                    _render_result(result)

            except Exception as e:
                error_msg = f"Não consegui responder: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_clarification" not in st.session_state:
    st.session_state.awaiting_clarification = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ── Header ─────────────────────────────────────────────────────────────────────
_logo_path = os.path.join(os.path.dirname(__file__), "logo.jpg")
_logo_b64 = base64.b64encode(open(_logo_path, "rb").read()).decode() if os.path.exists(_logo_path) else ""
_logo_html = f'<img src="data:image/jpeg;base64,{_logo_b64}" style="height:40px;border-radius:8px;" />' if _logo_b64 else '<span class="sz-badge">Seazone</span>'

st.markdown(f"""
<div class="sz-header">
  {_logo_html}
  <span class="sz-title">Data Agent</span>
</div>
<p class="sz-subtitle">Faça perguntas sobre negócio e tenha os resultados descomplicados 😊</p>
<hr class="sz-divider"/>
""", unsafe_allow_html=True)

# ── Tela de boas-vindas ────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown('<div class="sz-section">Sugestões para começar</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    for i, (icon, label, question) in enumerate(SUGGESTIONS):
        with [col1, col2, col3][i % 3]:
            if st.button(f"{icon} **{label}**\n\n{question}", key=f"sug_{i}"):
                st.session_state.pending_question = question
                st.rerun()
    st.markdown("<br/>", unsafe_allow_html=True)

# ── Histórico de mensagens ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        _render_user_msg(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            if msg.get("chart") is not None:
                st.plotly_chart(msg["chart"], use_container_width=True)
            if msg.get("dataframe") is not None:
                st.dataframe(msg["dataframe"], use_container_width=True, hide_index=True)
            if msg.get("sql"):
                with st.expander("🔍 SQL gerado"):
                    if msg.get("tables"):
                        st.caption("Tabelas: " + " · ".join(f"`{t}`" for t in msg["tables"]))
                    st.code(msg["sql"], language="sql")

# ── Processar sugestão clicada ─────────────────────────────────────────────────
if st.session_state.pending_question:
    q = st.session_state.pending_question
    st.session_state.pending_question = None
    if st.session_state.awaiting_clarification:
        _handle_clarification(q)
    else:
        _process_question(q)
        st.rerun()

# ── Input de texto ─────────────────────────────────────────────────────────────
question = st.chat_input("Faça uma pergunta sobre o negócio...")

if question:
    if st.session_state.awaiting_clarification:
        _handle_clarification(question)
    else:
        _process_question(question)
        st.rerun()
