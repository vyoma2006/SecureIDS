"""
Reusable Plotly chart builders for the dashboard.

Every function takes plain dicts/lists (exactly what api_client.py returns —
already-parsed JSON) and returns a plotly.graph_objects.Figure. No
Streamlit calls in here, so these are easy to unit test / reuse.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BRAND_COLORWAY = px.colors.qualitative.Set2

BENIGN_COLOR = "#2E9E6B"
ATTACK_COLOR = "#D64545"
ACCENT = "#2E86AB"


def _style(fig: go.Figure, height: int = 380) -> go.Figure:
    """One consistent look for every chart: clean white template, tight margins."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(size=13),
        title=dict(font=dict(size=16)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def confusion_matrix_heatmap(confusion_matrix: list[list[int]], classes: list[str]) -> go.Figure:
    """Confusion matrix as an annotated heatmap. Rows=true, cols=predicted."""
    fig = px.imshow(
        confusion_matrix,
        x=classes,
        y=classes,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="True label", color="Count"),
    )
    fig.update_layout(
        title="Confusion Matrix — Baseline Model",
        xaxis_title="Predicted label",
        yaxis_title="True label",
    )
    return _style(fig, height=460)


def per_class_metrics_bar(per_class: dict[str, dict]) -> go.Figure:
    """Grouped bar chart: precision / recall / F1 for each class."""
    rows = []
    for class_name, metrics in per_class.items():
        for metric_name in ("precision", "recall", "f1"):
            rows.append({"class": class_name, "metric": metric_name, "value": metrics[metric_name]})
    df = pd.DataFrame(rows)

    fig = px.bar(
        df,
        x="class",
        y="value",
        color="metric",
        barmode="group",
        color_discrete_sequence=BRAND_COLORWAY,
        labels={"value": "Score", "class": "Class"},
        title="Per-Class Precision / Recall / F1 — Baseline Model",
    )
    fig.update_yaxes(range=[0, 1])
    return _style(fig)


def evasion_rate_by_epsilon(evasion_results: list[dict], metric: str = "evasion_rate") -> go.Figure:
    """
    Line chart: `metric` vs. epsilon, one line per attack class.
    evasion_results is the `results` list from /metrics/evasion-rate.
    `metric` is either "evasion_rate" (default) or "confidence_drop".
    """
    df = pd.DataFrame(evasion_results).sort_values(["attack_class", "epsilon"])

    metric_label = "Evasion rate" if metric == "evasion_rate" else "Confidence drop"
    fig = px.line(
        df,
        x="epsilon",
        y=metric,
        color="attack_class",
        markers=True,
        color_discrete_sequence=BRAND_COLORWAY,
        labels={"epsilon": "Perturbation budget (ε)", metric: metric_label, "attack_class": "Class"},
        title=f"FGSM {metric_label} by Epsilon",
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.update_layout(legend_title_text="Attack class")
    return _style(fig)


def before_after_comparison(baseline: dict, robust: dict) -> go.Figure:
    """
    Grouped bar chart comparing baseline vs. robust model on
    accuracy / detection_rate / macro_f1.
    """
    metrics = ["accuracy", "detection_rate", "macro_f1"]
    df = pd.DataFrame(
        {
            "metric": metrics * 2,
            "value": [baseline[m] for m in metrics] + [robust[m] for m in metrics],
            "model": ["Baseline"] * len(metrics) + ["Adversarially Trained"] * len(metrics),
        }
    )

    fig = px.bar(
        df,
        x="metric",
        y="value",
        color="model",
        barmode="group",
        color_discrete_sequence=["#8AA1B1", "#2E86AB"],
        labels={"value": "Score", "metric": "Metric"},
        title="Before vs. After Adversarial Training",
    )
    fig.update_yaxes(range=[0, 1])
    return _style(fig)


# ---------------------------------------------------------------------------
# Live prediction demo
# ---------------------------------------------------------------------------

def class_color(class_name: str) -> str:
    """Green for Benign, red for everything else — the one distinction that matters at a glance."""
    return BENIGN_COLOR if class_name.strip().lower() == "benign" else ATTACK_COLOR


def prediction_summary_bar(class_counts: dict[str, int]) -> go.Figure:
    """Horizontal bar: how many uploaded rows were classified into each class."""
    items = sorted(class_counts.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    values = [v for _, v in items]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=[class_color(n) for n in names],
            text=values,
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(title="Predicted classes", xaxis_title="Rows", showlegend=False)
    return _style(fig, height=max(240, 60 + 38 * len(names)))


def probability_bar(probabilities: dict[str, float], predicted_class: str) -> go.Figure:
    """Horizontal bar of the model's full probability distribution for ONE row."""
    items = sorted(probabilities.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    values = [v for _, v in items]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=[class_color(n) if n == predicted_class else "#C9D3DB" for n in names],
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_xaxes(range=[0, 1.12], tickformat=".0%")
    fig.update_layout(title=f"Model confidence for this row → {predicted_class}", showlegend=False)
    return _style(fig, height=360)
