"""Reusable presentation primitives (cards, KPIs, gauges, badges, tables)."""
from __future__ import annotations

import html
import itertools
from typing import Iterable, List, Optional, Sequence

import pandas as pd
import streamlit as st

_card_ids = itertools.count(1)


def esc(x: object) -> str:
    return html.escape(str(x if x is not None else ""))


def stretch(fn, *args, **kwargs):
    """Call a Streamlit element full-width across API generations (width= vs use_container_width=)."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def card(**kwargs):
    """A bordered container that works on any Streamlit version that supports ``border``."""
    try:
        return st.container(border=True, key=f"card_{next(_card_ids)}", **kwargs)
    except TypeError:
        return st.container(border=True, **kwargs)


def html_block(markup: str) -> None:
    st.markdown(markup.replace("\n", ""), unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    html_block(f'<div class="section-title">{esc(title)}</div>' + (f'<div class="section-sub">{esc(subtitle)}</div>' if subtitle else ""))


def hero(title: str, tagline: str, pill: str) -> None:
    html_block(f'<div class="hero"><div class="hero-logo">🧭</div><div><h1>{esc(title)}</h1><p>{esc(tagline)}</p></div>'
               f'<div class="hero-pill">{esc(pill)}</div></div>')


def kpi(label: str, value: str, sub: str = "", icon: str = "📊", tone: str = "blue") -> None:
    html_block(f'<div class="kpi kpi-{tone}"><div class="kpi-icon">{icon}</div><div><div class="kpi-label">{esc(label)}</div>'
               f'<div class="kpi-value">{esc(value)}</div><div class="kpi-sub">{esc(sub)}</div></div></div>')


def badge(text: str, kind: Optional[str] = None) -> str:
    kind = (kind or text).lower()
    return f'<span class="badge b-{esc(kind)}">{esc(text)}</span>'


def empty_state(icon: str, title: str, message: str, steps: Sequence[str] = ()) -> None:
    steps_html = "".join(f'<div class="step"><b>{i}.</b> {esc(s)}</div>' for i, s in enumerate(steps, 1))
    html_block(f'<div class="empty"><div class="big">{icon}</div><h4>{esc(title)}</h4><div>{esc(message)}</div>'
               + (f'<div class="steps">{steps_html}</div>' if steps_html else "") + "</div>")


def gauge(score: Optional[float], label: str, color: str = "#94a3b8", size: int = 240) -> None:
    arc = 251.3
    val = 0 if score is None else max(0, min(100, score))
    shown = "—" if score is None else f"{score:.0f}"
    html_block(
        f'<div class="gauge-wrap"><svg width="{size}" viewBox="0 0 200 125" role="img" aria-label="health gauge">'
        f'<path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#e8ebf3" stroke-width="16" stroke-linecap="round"/>'
        f'<path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round" '
        f'stroke-dasharray="{val / 100 * arc:.1f} {arc}"/>'
        f'<text x="100" y="92" text-anchor="middle" font-size="40" font-weight="800" fill="#0f172a">{shown}</text>'
        f'<text x="100" y="114" text-anchor="middle" font-size="11" fill="#64748b">out of 100</text></svg>'
        f'<div class="gauge-label" style="color:{color}">{esc(label)}</div></div>')


def dimension_bar(label: str, score: Optional[float], note: str = "", weight: Optional[float] = None) -> None:
    pct = 0 if score is None else max(0, min(100, score))
    color = "#cbd5e1" if score is None else "#16a34a" if pct >= 80 else "#d97706" if pct >= 60 else "#ea580c" if pct >= 40 else "#dc2626"
    right = "—" if score is None else f"{pct:.0f}/100"
    w = f' <span class="muted" style="font-weight:500">· weight {int(weight * 100)}%</span>' if weight else ""
    html_block(f'<div class="dim"><div class="dim-top"><span>{esc(label)}{w}</span><span>{right}</span></div>'
               f'<div class="bar"><span style="width:{pct}%;background:{color}"></span></div>'
               + (f'<div class="dim-note">{esc(note)}</div>' if note else "") + "</div>")


def row_items(rows: Iterable[tuple], empty_msg: str = "Nothing to show yet.") -> None:
    """rows: (badge_html, main_text, meta_text)"""
    out = []
    for b, main, meta in rows:
        out.append(f'<div class="row-item"><div>{b}</div><div class="row-main">{esc(main)}'
                   f'<div class="row-meta">{esc(meta)}</div></div></div>')
    html_block("".join(out) if out else f'<div class="muted" style="padding:10px 0">{esc(empty_msg)}</div>')


def heat_matrix(counts: dict) -> None:
    """counts[(probability, impact)] -> number of risks."""
    def colour(score: int) -> str:
        return "#fecaca" if score >= 16 else "#fed7aa" if score >= 10 else "#fef08a" if score >= 5 else "#bbf7d0"
    rows = []
    for imp in range(5, 0, -1):
        cells = "".join(
            f'<td style="background:{colour(p * imp)}">{counts.get((p, imp), "") or ""}</td>' for p in range(1, 6))
        rows.append(f"<tr><th>{imp}</th>{cells}</tr>")
    head = "<tr><th></th>" + "".join(f"<th>{p}</th>" for p in range(1, 6)) + "</tr>"
    html_block(f'<table class="heat">{"".join(rows)}{head}</table>'
               '<div class="muted" style="text-align:center;font-size:.72rem;margin-top:4px">Probability → · Impact ↑</div>')


def show_df(df: pd.DataFrame, height: Optional[int] = None, **kwargs) -> None:
    if df is None or df.empty:
        st.caption("No records.")
        return
    kw = dict(hide_index=True, **kwargs)
    if height:
        kw["height"] = height
    stretch(st.dataframe, df, **kw)


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def bar_chart(df: pd.DataFrame, x: str, y: str, color: str = "#4f46e5", height: int = 220) -> None:
    if df is None or df.empty:
        st.caption("No data yet.")
        return
    try:
        import altair as alt
        chart = (alt.Chart(df).mark_bar(cornerRadiusEnd=5, color=color, size=22)
                 .encode(y=alt.Y(f"{x}:N", sort="-x", title=None), x=alt.X(f"{y}:Q", title=None, axis=alt.Axis(tickMinStep=1)),
                         tooltip=[x, y]).properties(height=height))
        stretch(st.altair_chart, chart)
    except Exception:
        st.bar_chart(df.set_index(x)[y], height=height)
