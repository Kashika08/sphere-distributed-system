#!/usr/bin/env python3
"""
Dash app that live‑plots actual CPU values against 13‑step‑ahead
predictions (average of the last three horizon points) with a 10‑second
step interval.

Files watched:
    ml_data.log         – "timestamp, actual, other"
    ml_predictions.log  – "timestamp, p1, p2, …, p15"
"""

import os
from pathlib import Path
from datetime import timedelta

import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# ── Configuration ──────────────────────────────────────────────────────────────
STEP_INTERVAL_SEC = 10        # seconds between successive horizon steps
TARGET_STEP       = 13        # we want the 13‑step forecast
OFFSET_SEC        = STEP_INTERVAL_SEC * TARGET_STEP

ACTUAL_LOG   = Path("ml_data.log")
PRED_LOG     = Path("ml_predictions.log")

# ── Helper functions ───────────────────────────────────────────────────────────
def load_actual(path: Path) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [timestamp, actual].
    """
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "actual"])

    df = pd.read_csv(
        path,
        header=None,
        names=["timestamp", "actual", "other"],
        parse_dates=["timestamp"],
        usecols=[0, 1],                 # ignore the "other" column
    )
    return df.sort_values("timestamp")


def load_predictions(path: Path) -> pd.DataFrame:
    """
    Reads prediction lines, calculates the average of the last three
    horizon points (columns 13‑15) and assigns it to timestamp + 130 s.
    Returns DataFrame [timestamp, predicted].
    """
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "predicted"])

    records = []
    with path.open() as fh:
        for raw in fh:
            parts = [p.strip() for p in raw.strip().split(",")]
            if len(parts) < 16:      # 1 timestamp + 15 preds required
                continue

            ts         = pd.to_datetime(parts[0])
            preds      = list(map(float, parts[1:]))
            avg_last3  = sum(preds[-3:]) / 3.0

            pred_ts = ts + timedelta(seconds=OFFSET_SEC)
            records.append({"timestamp": pred_ts, "predicted": avg_last3})

    df = pd.DataFrame(records)
    return df.sort_values("timestamp")


def merge_series(df_actual: pd.DataFrame, df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Nearest‑timestamp merge (tolerance ±5 s) so small jitter in log times
    isn’t fatal.
    """
    if df_actual.empty or df_pred.empty:
        return pd.DataFrame(columns=["timestamp", "actual", "predicted"])

    # make sure both are datetime64 & sorted
    df_actual  = df_actual.copy().sort_values("timestamp")
    df_pred    = df_pred.copy().sort_values("timestamp")

    merged = pd.merge_asof(
        df_pred,                       # left
        df_actual,                     # right
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=5),
    )
    merged.dropna(subset=["actual"], inplace=True)
    return merged


# ── Dash layout ────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.title = "Live Actual vs Forecast"

app.layout = html.Div(
    [
        html.H3("Actual vs Predicted (13‑step horizon, 10‑s interval)"),
        dcc.Graph(id="live‑graph", config={"displayModeBar": False}),
        dcc.Interval(id="refresh", interval=5_000, n_intervals=0),  # every 5 s
        html.Div(id="metrics", style={"marginTop": "1rem"}),
    ],
    style={"width": "90%", "margin": "auto", "fontFamily": "Arial, sans‑serif"},
)

# ── Callback ───────────────────────────────────────────────────────────────────
@app.callback(
    [Output("live‑graph", "figure"), Output("metrics", "children")],
    [Input("refresh", "n_intervals")],
)
def update_graph(_):
    # Reload logs every tick (cheap for < few MB files)
    df_actual = load_actual(ACTUAL_LOG)
    df_pred   = load_predictions(PRED_LOG)
    df        = merge_series(df_actual, df_pred)

    # Build Plotly figure
    traces = []
    if not df.empty:
        traces.append(
            go.Scatter(
                x=df["timestamp"],
                y=df["actual"],
                mode="lines+markers",
                name="Actual",
            )
        )
        traces.append(
            go.Scatter(
                x=df["timestamp"],
                y=df["predicted"],
                mode="lines+markers",
                name="Predicted (avg last 3)",
            )
        )

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis_title="Time",
            yaxis_title="CPU %",
            legend=dict(orientation="h", y=-0.25),
        ),
    )

    # Simple error metric
    mae_text = ""
    # if not df.empty:
    #     mae = (df["actual"] - df["predicted"]).abs().mean()
    #     mae_text = f"Current MAE over {len(df)} points: **{mae:0.3f}**"

    return fig, dcc.Markdown(mae_text)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Ensure logs exist so the user sees something immediately
    for f in (ACTUAL_LOG, PRED_LOG):
        if not f.exists():
            print(f"⚠️  {f} not found. Waiting for file to appear …")

    app.run(debug=True, host="0.0.0.0", port=8050)
