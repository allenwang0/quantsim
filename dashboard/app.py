"""
QuantSim Dashboard v2 — 5-tab Streamlit app.

Runs as a SEPARATE OS PROCESS from the trading engine.
Reads from SQLite every 2s. Never writes. Never blocks the trading loop.

Tabs:
  1. Portfolio Overview   — equity curve, drawdown, 4 key metrics
  2. Positions & Greeks  — positions table, aggregate Greeks, fills
  3. Strategy Analytics  — per-strategy P&L, rolling Sharpe, monthly heatmap
  4. Research            — WFO results, ML IC history, GARCH vol forecasts
  5. Risk Monitor        — circuit breaker status, alerts, Greeks limits

Run:  streamlit run dashboard/app.py
"""

import os, json, sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="QuantSim",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DB_PATH = os.environ.get("QUANTSIM_DB", os.path.expanduser("~/.quantsim/quantsim.db"))
REFRESH = 2

# ── Colors ─────────────────────────────────────────────────────────────────────
C = {
    "bg": "#08080f", "surface": "#111118", "surface2": "#18181f",
    "border": "#252535", "accent": "#00d4ff", "accent2": "#7b61ff",
    "green": "#00e676", "red": "#ff3d6b", "yellow": "#ffd740",
    "text": "#e0e0f0", "muted": "#5a5a7a",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@600;700;800&display=swap');
.stApp {{ background:{C['bg']}; color:{C['text']}; font-family:'JetBrains Mono',monospace; }}
.main .block-container {{ padding:1.5rem 2rem; max-width:100%; }}
h1,h2,h3 {{ font-family:'Syne',sans-serif!important; letter-spacing:-0.02em; }}
.mc {{ background:{C['surface']}; border:1px solid {C['border']}; border-radius:8px;
       padding:1.1rem 1.4rem; position:relative; overflow:hidden; margin-bottom:0.5rem; }}
.mc::before {{ content:''; position:absolute; top:0; left:0; right:0; height:2px;
               background:linear-gradient(90deg,{C['accent']},{C['accent2']}); }}
.ml {{ font-size:0.65rem; color:{C['muted']}; text-transform:uppercase; letter-spacing:0.12em; }}
.mv {{ font-family:'Syne',sans-serif; font-size:1.7rem; font-weight:700; line-height:1; }}
.ms {{ font-size:0.7rem; color:{C['muted']}; margin-top:0.25rem; }}
.pos {{ color:{C['green']}!important; }}
.neg {{ color:{C['red']}!important; }}
.sh {{ font-size:0.68rem; color:{C['muted']}; text-transform:uppercase; letter-spacing:0.12em;
       border-bottom:1px solid {C['border']}; padding-bottom:0.4rem; margin-bottom:0.8rem; }}
.alert {{ background:rgba(255,61,107,0.12); border:1px solid {C['red']};
          border-radius:6px; padding:0.7rem 1rem; margin-bottom:0.8rem; font-size:0.8rem; }}
.warn {{ background:rgba(255,215,64,0.10); border:1px solid {C['yellow']};
         border-radius:6px; padding:0.7rem 1rem; margin-bottom:0.8rem; font-size:0.8rem; }}
div[data-testid="stDataFrame"] {{ background:{C['surface']}; border-radius:8px; border:1px solid {C['border']}; }}
.stTabs [data-baseweb="tab"] {{ font-family:'JetBrains Mono',monospace; font-size:0.72rem;
                                 color:{C['muted']}; text-transform:uppercase; letter-spacing:0.1em; }}
.stTabs [aria-selected="true"] {{ color:{C['accent']}!important; }}
</style>"""

PLOT_LAYOUT = dict(
    paper_bgcolor=C["bg"], plot_bgcolor=C["surface"],
    font=dict(family="JetBrains Mono", color=C["text"], size=11),
    margin=dict(l=0, r=0, t=10, b=0),
    hovermode="x unified",
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor=C["border"], zeroline=False),
)

# ── DB helpers ─────────────────────────────────────────────────────────────────

def _conn():
    if not os.path.exists(DB_PATH): return None
    c = sqlite3.connect(DB_PATH, timeout=3)
    c.row_factory = sqlite3.Row
    return c

def _q(sql, params=()):
    c = _conn()
    if not c: return []
    try:
        return [dict(r) for r in c.execute(sql, params).fetchall()]
    except Exception: return []
    finally: c.close()

@st.cache_data(ttl=REFRESH)
def load_snapshot():
    rows = _q("SELECT payload FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1")
    if rows:
        try: return json.loads(rows[0]["payload"])
        except: pass
    return {}

@st.cache_data(ttl=REFRESH)
def load_equity_curve(days=730):
    cutoff = int((datetime.utcnow()-timedelta(days=days)).timestamp())
    rows = _q("SELECT timestamp,total_equity FROM portfolio_snapshots WHERE timestamp>=? ORDER BY timestamp ASC", (cutoff,))
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df.set_index("dt")

@st.cache_data(ttl=REFRESH)
def load_trades(n=200):
    rows = _q(f"SELECT * FROM trades ORDER BY entry_timestamp DESC LIMIT {n}")
    return pd.DataFrame(rows) if rows else pd.DataFrame()

@st.cache_data(ttl=REFRESH)
def load_alerts():
    return _q("SELECT * FROM risk_alerts WHERE dismissed=0 ORDER BY timestamp DESC LIMIT 10")

@st.cache_data(ttl=REFRESH)
def load_strategy_perf():
    rows = _q("SELECT * FROM strategy_performance ORDER BY timestamp DESC LIMIT 2000")
    return pd.DataFrame(rows) if rows else pd.DataFrame()

@st.cache_data(ttl=REFRESH)
def load_wfo_results():
    rows = _q("SELECT * FROM wfo_results ORDER BY run_timestamp DESC LIMIT 200")
    return pd.DataFrame(rows) if rows else pd.DataFrame()

@st.cache_data(ttl=REFRESH)
def load_ml_runs():
    rows = _q("SELECT * FROM ml_model_runs ORDER BY trained_at DESC LIMIT 100")
    return pd.DataFrame(rows) if rows else pd.DataFrame()

@st.cache_data(ttl=REFRESH)
def load_ml_ic():
    rows = _q("SELECT * FROM ml_ic_history ORDER BY eval_date DESC LIMIT 500")
    return pd.DataFrame(rows) if rows else pd.DataFrame()

@st.cache_data(ttl=REFRESH)
def load_garch_forecasts(asset_id=None):
    if asset_id:
        rows = _q("SELECT * FROM garch_forecasts WHERE asset_id=? ORDER BY timestamp DESC LIMIT 500", (asset_id,))
    else:
        rows = _q("SELECT * FROM garch_forecasts ORDER BY timestamp DESC LIMIT 2000")
    return pd.DataFrame(rows) if rows else pd.DataFrame()

@st.cache_data(ttl=REFRESH)
def load_greeks_log():
    rows = _q("SELECT * FROM options_greeks_log ORDER BY timestamp DESC LIMIT 500")
    return pd.DataFrame(rows) if rows else pd.DataFrame()

# ── Chart helpers ───────────────────────────────────────────────────────────────

def chart_equity(eq_df):
    if eq_df.empty:
        fig = go.Figure()
        fig.update_layout(**PLOT_LAYOUT, height=320)
        fig.add_annotation(text="No data yet", x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(color=C["muted"]))
        return fig
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.68, 0.32], vertical_spacing=0.02)
    fig.add_trace(go.Scatter(
        x=eq_df.index, y=eq_df["total_equity"], mode="lines", name="Equity",
        line=dict(color=C["accent"], width=2),
        fill="tozeroy", fillcolor=f"rgba(0,212,255,0.04)"), row=1, col=1)

    eq_df = eq_df.copy()
    eq_df["peak"] = eq_df["total_equity"].expanding().max()
    eq_df["dd"] = (eq_df["total_equity"] - eq_df["peak"]) / eq_df["peak"] * 100
    fig.add_trace(go.Scatter(
        x=eq_df.index, y=eq_df["dd"], mode="lines", name="Drawdown %",
        line=dict(color=C["red"], width=1.5),
        fill="tozeroy", fillcolor=f"rgba(255,61,107,0.13)"), row=2, col=1)

    layout = dict(**PLOT_LAYOUT, height=360, showlegend=False)
    layout["yaxis"] = dict(showgrid=True, gridcolor=C["border"], zeroline=False, tickprefix="$", tickformat=",.0f")
    layout["yaxis2"] = dict(showgrid=True, gridcolor=C["border"], zeroline=True, zerolinecolor=C["muted"], ticksuffix="%")
    fig.update_layout(**layout)
    return fig

def chart_monthly_heatmap(eq_df):
    if eq_df.empty or len(eq_df) < 25: return go.Figure()
    monthly = eq_df["total_equity"].resample("ME").last()
    ret = monthly.pct_change(fill_method=None).dropna() * 100
    if ret.empty: return go.Figure()
    pivot = ret.groupby([ret.index.year, ret.index.month]).first().unstack(1)
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    mx = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.01) if pivot.size else 5
    ann = []
    for i,yr in enumerate(pivot.index):
        for j,mo in enumerate(pivot.columns):
            v = pivot.iloc[i, j]
            if not np.isnan(v):
                ann.append(dict(x=j, y=i, text=f"{v:.1f}%",
                    font=dict(size=9, color="white" if abs(v) > mx*0.4 else C["muted"]),
                    xref="x", yref="y", showarrow=False))
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns),
        y=[str(yr) for yr in pivot.index],
        colorscale=[[0,"#cc1f3d"],[0.5,C["surface2"]],[1,"#00e676"]],
        zmin=-mx, zmax=mx, showscale=False))
    fig.update_layout(**PLOT_LAYOUT, height=max(120, len(pivot)*30+50),
                      annotations=ann, xaxis=dict(side="top"))
    return fig

def chart_rolling_sharpe(eq_df, window=252):
    if eq_df.empty or len(eq_df) < 30: return go.Figure()
    r = eq_df["total_equity"].pct_change(fill_method=None).dropna()
    rs = (r.rolling(min(window, len(r)//2)).mean() /
          r.rolling(min(window, len(r)//2)).std() * np.sqrt(252)).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rs.index, y=rs, mode="lines",
                             line=dict(color=C["accent2"], width=2)))
    fig.add_hline(y=0, line_color=C["muted"], line_dash="dash", line_width=1)
    fig.add_hline(y=1, line_color=C["green"], line_dash="dot", line_width=1)
    fig.update_layout(**PLOT_LAYOUT, height=180, showlegend=False)
    return fig

def chart_strategy_pnl(strat_df):
    if strat_df.empty: return go.Figure()
    if "strategy_id" not in strat_df.columns or "daily_pnl" not in strat_df.columns:
        return go.Figure()
    totals = strat_df.groupby("strategy_id")["daily_pnl"].sum().sort_values()
    fig = go.Figure(go.Bar(
        x=totals.values, y=totals.index, orientation="h",
        marker_color=[C["green"] if v >= 0 else C["red"] for v in totals],
        text=[f"${v:+,.0f}" for v in totals], textposition="outside",
        textfont=dict(size=11, color=C["text"])))
    fig.update_layout(**PLOT_LAYOUT, height=max(180, len(totals)*40+60),
                      xaxis=dict(tickprefix="$", showgrid=True, gridcolor=C["border"]))
    return fig

def chart_wfo_oos(wfo_df):
    if wfo_df.empty or "oos_sharpe" not in wfo_df.columns: return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(wfo_df))), y=wfo_df["oos_sharpe"],
        marker_color=[C["green"] if v > 0 else C["red"] for v in wfo_df["oos_sharpe"]],
        name="OOS Sharpe"))
    if "is_sharpe" in wfo_df.columns:
        fig.add_trace(go.Scatter(
            x=list(range(len(wfo_df))), y=wfo_df["is_sharpe"],
            mode="lines+markers", name="IS Sharpe",
            line=dict(color=C["yellow"], dash="dot", width=1.5)))
    fig.add_hline(y=0, line_color=C["muted"], line_dash="dash", line_width=1)
    fig.update_layout(**PLOT_LAYOUT, height=240,
                      xaxis_title="Window", yaxis_title="Sharpe")
    return fig

def chart_ic_history(ic_df):
    if ic_df.empty or "ic" not in ic_df.columns: return go.Figure()
    ic_df = ic_df.copy()
    ic_df["dt"] = pd.to_datetime(ic_df["eval_date"], unit="s", utc=True)
    ic_df = ic_df.sort_values("dt")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ic_df["dt"], y=ic_df["ic"], mode="lines",
        line=dict(color=C["accent"], width=1.5), name="IC"))
    fig.add_trace(go.Scatter(
        x=ic_df["dt"], y=ic_df["ic"].rolling(12).mean(),
        mode="lines", line=dict(color=C["yellow"], width=2, dash="dot"), name="12-period MA"))
    fig.add_hline(y=0.02, line_color=C["green"], line_dash="dot",
                  annotation_text="min IC threshold", line_width=1)
    fig.add_hline(y=0, line_color=C["muted"], line_dash="dash", line_width=1)
    fig.update_layout(**PLOT_LAYOUT, height=200)
    return fig

def chart_garch_vol(garch_df):
    if garch_df.empty or "forecast_vol" not in garch_df.columns: return go.Figure()
    garch_df = garch_df.copy()
    garch_df["dt"] = pd.to_datetime(garch_df["timestamp"], unit="s", utc=True)
    garch_df = garch_df.sort_values("dt")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=garch_df["dt"], y=garch_df["forecast_vol"] * 100,
        mode="lines", line=dict(color=C["accent2"], width=2), name="GARCH Forecast Vol %"))
    if "realized_vol" in garch_df.columns:
        fig.add_trace(go.Scatter(
            x=garch_df["dt"], y=garch_df["realized_vol"] * 100,
            mode="lines", line=dict(color=C["muted"], width=1, dash="dot"), name="Realized Vol %"))
    fig.update_layout(**PLOT_LAYOUT, height=200, yaxis_title="Annualized Vol %")
    return fig

def mc(label, value, sub="", positive=None):
    cls = " pos" if positive is True else (" neg" if positive is False else "")
    return f"""<div class="mc"><div class="ml">{label}</div>
               <div class="mv{cls}">{value}</div>
               {"<div class='ms'>"+sub+"</div>" if sub else ""}</div>"""

# ── Main app ────────────────────────────────────────────────────────────────────

def main():
    st.markdown(CSS, unsafe_allow_html=True)

    # Header
    c1, c2 = st.columns([3,1])
    with c1:
        st.markdown("<h1 style='font-family:Syne;font-size:1.6rem;margin:0;'>◈ QUANTSIM v2</h1>"
                    f"<p style='color:{C['muted']};font-size:0.65rem;margin:0;letter-spacing:0.12em;'>"
                    "EVENT-DRIVEN · VECTORIZED · ML · PAPER TRADING</p>", unsafe_allow_html=True)
    with c2:
        status = "● LIVE" if os.path.exists(DB_PATH) else "✕ NO DB"
        color = C["green"] if os.path.exists(DB_PATH) else C["red"]
        st.markdown(f"<div style='text-align:right;font-size:0.7rem;color:{color};margin-top:0.6rem;'>"
                    f"{status}</div>", unsafe_allow_html=True)

    st.markdown(f"<hr style='border-color:{C['border']};margin:0.6rem 0;'>", unsafe_allow_html=True)

    # Alerts banner
    alerts = load_alerts()
    for a in alerts[:2]:
        sev = a.get("severity","WARNING")
        cls = "alert" if sev == "CRITICAL" else "warn"
        icon = "⛔" if sev == "CRITICAL" else "⚠"
        st.markdown(f"<div class='{cls}'>{icon} <strong>{a.get('risk_type','')}</strong>: {a.get('message','')}</div>",
                    unsafe_allow_html=True)

    snapshot = load_snapshot()
    eq_df = load_equity_curve()

    # ── Tab layout ────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "PORTFOLIO", "POSITIONS", "STRATEGY", "RESEARCH", "RISK"
    ])

    # ── Tab 1: Portfolio Overview ─────────────────────────────────────────────
    with t1:
        if snapshot:
            equity = snapshot.get("total_equity", 0)
            dd = snapshot.get("current_drawdown", 0)
            rpnl = snapshot.get("realized_pnl", 0)
            upnl = snapshot.get("unrealized_pnl", 0)

            today_pnl = 0
            if not eq_df.empty and len(eq_df) >= 2:
                today_pnl = eq_df["total_equity"].iloc[-1] - eq_df["total_equity"].iloc[-2]
            ytd = 0
            if not eq_df.empty:
                yr = eq_df[eq_df.index.year == datetime.utcnow().year]
                if len(yr) >= 2:
                    ytd = (yr["total_equity"].iloc[-1] / yr["total_equity"].iloc[0]) - 1

            c1,c2,c3,c4 = st.columns(4)
            with c1: st.markdown(mc("Total Equity", f"${equity:,.0f}"), unsafe_allow_html=True)
            with c2: st.markdown(mc("Today P&L", f"${today_pnl:+,.0f}",
                                    positive=today_pnl>=0), unsafe_allow_html=True)
            with c3: st.markdown(mc("Drawdown", f"{dd:.2%}", positive=dd>=-0.05),
                                 unsafe_allow_html=True)
            with c4: st.markdown(mc("YTD", f"{ytd:+.2%}", positive=ytd>=0),
                                 unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div class='sh'>Equity & Drawdown</div>", unsafe_allow_html=True)
            st.plotly_chart(chart_equity(eq_df), use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.info("No portfolio data. Run a backtest: `python scripts/run_backtest.py --strategy sma --symbol SPY`")

    # ── Tab 2: Positions & Greeks ─────────────────────────────────────────────
    with t2:
        positions = (snapshot or {}).get("positions", {})
        if positions:
            rows = []
            for k, p in positions.items():
                pnl = p.get("unrealized_pnl", 0)
                rows.append({
                    "Symbol": p.get("asset_id", k),
                    "Type": p.get("asset_type",""),
                    "Qty": p.get("quantity",0),
                    "Avg Cost": f"${p.get('average_cost',0):.2f}",
                    "Current": f"${p.get('current_price',0):.2f}",
                    "Unrlzd P&L": f"${pnl:+,.0f}",
                    "Strategy": p.get("strategy_id",""),
                    "Δ": f"{p.get('delta',0):.3f}",
                    "θ": f"{p.get('theta',0):.4f}",
                    "ν": f"{p.get('vega',0):.4f}",
                })
            st.markdown(f"<div class='sh'>Open Positions ({len(rows)})</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Aggregate Greeks
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div class='sh'>Portfolio Greeks</div>", unsafe_allow_html=True)
            total_delta = sum(p.get("delta",0)*p.get("quantity",0)*(100 if p.get("asset_type")=="OPTION" else 1)
                              for p in positions.values())
            total_vega  = sum(p.get("vega",0)*p.get("quantity",0)*100
                              for p in positions.values() if p.get("asset_type")=="OPTION")
            total_theta = sum(p.get("theta",0)*p.get("quantity",0)*100
                              for p in positions.values() if p.get("asset_type")=="OPTION")
            gc1,gc2,gc3 = st.columns(3)
            with gc1: st.markdown(mc("Δ Delta", f"{total_delta:.2f}", "delta-equiv shares"), unsafe_allow_html=True)
            with gc2: st.markdown(mc("ν Vega",  f"${total_vega:.0f}", "$ per 1% IV"), unsafe_allow_html=True)
            with gc3: st.markdown(mc("θ Theta", f"${total_theta:.2f}", "$ per day"), unsafe_allow_html=True)

            # Greeks time series
            greeks_df = load_greeks_log()
            if not greeks_df.empty:
                greeks_df["dt"] = pd.to_datetime(greeks_df["timestamp"], unit="s", utc=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=greeks_df["dt"], y=greeks_df["portfolio_delta"],
                                         mode="lines", name="Delta", line=dict(color=C["accent"])))
                fig.update_layout(**PLOT_LAYOUT, height=180, title_text="Portfolio Delta History")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No open positions.")

        # Recent trades
        trades_df = load_trades()
        if not trades_df.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div class='sh'>Recent Trades</div>", unsafe_allow_html=True)
            display = trades_df[["asset_id","direction","quantity",
                                  "entry_price","exit_price","realized_pnl",
                                  "strategy_id","holding_bars"]].copy()
            display.columns = ["Symbol","Dir","Qty","Entry","Exit","P&L","Strategy","Bars"]
            display["P&L"] = display["P&L"].apply(lambda x: f"${x:+,.0f}" if pd.notna(x) else "Open")
            st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Tab 3: Strategy Analytics ─────────────────────────────────────────────
    with t3:
        strat_df = load_strategy_perf()
        if not strat_df.empty:
            st.markdown(f"<div class='sh'>P&L by Strategy</div>", unsafe_allow_html=True)
            st.plotly_chart(chart_strategy_pnl(strat_df), use_container_width=True,
                            config={"displayModeBar": False})
        else:
            trades_df2 = load_trades()
            if not trades_df2.empty and "realized_pnl" in trades_df2.columns:
                closed = trades_df2[trades_df2["realized_pnl"].notna()].copy()
                if not closed.empty and "strategy_id" in closed.columns:
                    fake_strat_df = closed.rename(columns={"realized_pnl": "daily_pnl"})
                    st.markdown(f"<div class='sh'>P&L by Strategy (from trades)</div>", unsafe_allow_html=True)
                    st.plotly_chart(chart_strategy_pnl(fake_strat_df), use_container_width=True,
                                    config={"displayModeBar": False})

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='sh'>Monthly Returns Heatmap</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_monthly_heatmap(eq_df), use_container_width=True,
                        config={"displayModeBar": False})

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='sh'>Rolling 12-Month Sharpe</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_rolling_sharpe(eq_df), use_container_width=True,
                        config={"displayModeBar": False})

    # ── Tab 4: Research ───────────────────────────────────────────────────────
    with t4:
        # WFO Results
        st.markdown(f"<div class='sh'>Walk-Forward Optimization Results</div>", unsafe_allow_html=True)
        wfo_df = load_wfo_results()
        if not wfo_df.empty:
            show_cols = [c for c in ["strategy_id","window_id","is_sharpe","oos_sharpe",
                                     "oos_return","oos_max_dd","best_params"] if c in wfo_df.columns]
            st.dataframe(wfo_df[show_cols].head(50), use_container_width=True, hide_index=True)
            st.plotly_chart(chart_wfo_oos(wfo_df), use_container_width=True,
                            config={"displayModeBar": False})

            # Summary stats
            if "oos_sharpe" in wfo_df.columns:
                avg_oos = wfo_df["oos_sharpe"].mean()
                win_rate = (wfo_df["oos_sharpe"] > 0).mean()
                mc1, mc2, mc3 = st.columns(3)
                with mc1: st.markdown(mc("Avg OOS Sharpe", f"{avg_oos:.3f}", positive=avg_oos>0), unsafe_allow_html=True)
                with mc2: st.markdown(mc("OOS Win Rate", f"{win_rate:.1%}", positive=win_rate>0.5), unsafe_allow_html=True)
                with mc3:
                    n_win = len(wfo_df)
                    st.markdown(mc("Windows Tested", str(n_win)), unsafe_allow_html=True)
        else:
            st.info("No WFO results yet. Run: `python scripts/run_wfo.py --signal sma --symbol SPY`")

        # ML IC History
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='sh'>ML Model IC History</div>", unsafe_allow_html=True)
        ic_df = load_ml_ic()
        if not ic_df.empty:
            st.plotly_chart(chart_ic_history(ic_df), use_container_width=True,
                            config={"displayModeBar": False})
            latest_ic = ic_df["ic"].iloc[0] if len(ic_df) > 0 else 0
            rolling_ic = ic_df["ic"].head(12).mean() if len(ic_df) >= 12 else ic_df["ic"].mean()
            icc1, icc2 = st.columns(2)
            with icc1: st.markdown(mc("Latest IC", f"{latest_ic:.4f}", positive=latest_ic > 0.02), unsafe_allow_html=True)
            with icc2: st.markdown(mc("Rolling 12-period IC", f"{rolling_ic:.4f}", positive=rolling_ic > 0.02), unsafe_allow_html=True)
        else:
            st.info("No ML IC data yet. ML strategy must be running.")

        # GARCH Vol Forecasts
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='sh'>GARCH Volatility Forecasts</div>", unsafe_allow_html=True)
        garch_df = load_garch_forecasts()
        if not garch_df.empty and "asset_id" in garch_df.columns:
            assets = garch_df["asset_id"].unique()[:5]
            selected = st.selectbox("Asset", assets, key="garch_asset") if len(assets) > 1 else assets[0]
            asset_garch = garch_df[garch_df["asset_id"] == selected]
            st.plotly_chart(chart_garch_vol(asset_garch), use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.info("No GARCH forecasts yet. GARCH runs automatically when enough price history exists.")

        # ML Training Runs
        ml_df = load_ml_runs()
        if not ml_df.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div class='sh'>ML Training Run History</div>", unsafe_allow_html=True)
            show_cols = [c for c in ["strategy_id","trained_at","n_samples","train_ic","val_ic"] if c in ml_df.columns]
            st.dataframe(ml_df[show_cols].head(20), use_container_width=True, hide_index=True)

    # ── Tab 5: Risk Monitor ────────────────────────────────────────────────────
    with t5:
        st.markdown(f"<div class='sh'>Active Alerts</div>", unsafe_allow_html=True)
        if alerts:
            for a in alerts:
                sev = a.get("severity","WARNING")
                icon = "⛔" if sev == "CRITICAL" else "⚠"
                cls = "alert" if sev == "CRITICAL" else "warn"
                ts = datetime.utcfromtimestamp(a.get("timestamp",0)).strftime("%Y-%m-%d %H:%M")
                st.markdown(f"<div class='{cls}'>{icon} [{ts}] <strong>{a.get('risk_type','')}</strong>: {a.get('message','')}</div>",
                            unsafe_allow_html=True)
        else:
            st.success("✓ No active risk alerts")

        # Portfolio metrics
        if snapshot:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div class='sh'>Portfolio Risk Metrics</div>", unsafe_allow_html=True)
            equity = snapshot.get("total_equity", 0)
            dd = snapshot.get("current_drawdown", 0)
            n_pos = len(snapshot.get("positions", {}))
            positions = snapshot.get("positions", {})
            max_pos_pct = max(
                (abs(p.get("market_value",0)) / equity if equity > 0 else 0)
                for p in positions.values()
            ) if positions else 0

            rc1,rc2,rc3,rc4 = st.columns(4)
            with rc1: st.markdown(mc("Current DD", f"{dd:.2%}",
                                     positive=dd>=-0.10), unsafe_allow_html=True)
            with rc2: st.markdown(mc("Open Positions", str(n_pos)), unsafe_allow_html=True)
            with rc3: st.markdown(mc("Max Position %", f"{max_pos_pct:.1%}",
                                     positive=max_pos_pct<=0.10), unsafe_allow_html=True)
            with rc4:
                halt_dd = -0.15
                close_dd = -0.20
                if dd <= close_dd:
                    status = "🔴 CLOSE ALL"
                elif dd <= halt_dd:
                    status = "🟡 HALT OPENS"
                else:
                    status = "🟢 NORMAL"
                st.markdown(mc("Circuit Breaker", status), unsafe_allow_html=True)

        # Greeks limits
        if snapshot and snapshot.get("positions"):
            positions = snapshot["positions"]
            equity = snapshot.get("total_equity", 1)
            total_delta = sum(p.get("delta",0)*p.get("quantity",0)*(100 if p.get("asset_type")=="OPTION" else 1)
                              for p in positions.values())
            total_vega  = sum(p.get("vega",0)*p.get("quantity",0)*100
                              for p in positions.values() if p.get("asset_type")=="OPTION")
            total_gamma = sum(p.get("gamma",0)*p.get("quantity",0)*100
                              for p in positions.values() if p.get("asset_type")=="OPTION")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div class='sh'>Greeks Limits (Options)</div>", unsafe_allow_html=True)
            LIMITS = {"Delta": (total_delta, 100.0), "Vega": (total_vega, 1000.0), "Gamma": (total_gamma, 50.0)}
            lc = st.columns(3)
            for i, (name, (val, limit)) in enumerate(LIMITS.items()):
                pct = abs(val) / limit if limit > 0 else 0
                color = C["red"] if pct > 0.8 else (C["yellow"] if pct > 0.5 else C["green"])
                status = "BREACHED" if pct > 1.0 else f"{pct:.0%} of limit"
                with lc[i]:
                    st.markdown(mc(f"{name} ({limit:.0f} limit)", f"{val:.2f}", status,
                                   positive=pct<=0.8), unsafe_allow_html=True)

    # Footer + auto-refresh
    st.markdown(
        f"<div style='text-align:center;font-size:0.62rem;color:{C['muted']};margin-top:2rem;'>"
        f"QuantSim v2 · DB: {DB_PATH.split('/')[-1]} · "
        f"{datetime.utcnow().strftime('%H:%M:%S')} UTC · refreshes every {REFRESH}s</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<script>setTimeout(()=>window.location.reload(),{REFRESH*1000});</script>",
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
