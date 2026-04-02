"""Smoke test: verifica que app.py importa sem erro e que build_chart não quebra."""
import pytest


def test_build_chart_bar_returns_figure():
    from app import build_chart
    import pandas as pd

    df = pd.DataFrame({"mes": ["2025-02", "2025-03"], "total": [100000, 120000]})
    fig = build_chart(df, chart_type="bar", chart_x="mes", chart_y="total", title="Teste")
    assert fig is not None


def test_build_chart_none_returns_none():
    from app import build_chart
    import pandas as pd

    df = pd.DataFrame({"mes": ["2025-02"], "total": [100000]})
    result = build_chart(df, chart_type="none", chart_x=None, chart_y=None, title=None)
    assert result is None
