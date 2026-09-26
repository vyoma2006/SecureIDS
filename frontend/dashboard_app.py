"""
SecureIDS Dashboard — Visualizer's main Streamlit app.

Run with (from the project root):
    streamlit run frontend/dashboard_app.py

Only talks to the FastAPI backend via api_client.py — never reads
results/metrics/*.json directly.
"""

import html
import io

import pandas as pd
import streamlit as st

import api_client
import plots

st.set_page_config(page_title="SecureIDS Dashboard", page_icon="\U0001F6E1\uFE0F", layout="wide")

# Light visual polish that works in both light and dark themes.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; max-width: 1200px; }
      [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.22);
        padding: 14px 18px;
        border-radius: 12px;
      }
      [data-testid="stSidebar"] .stRadio label { padding: 2px 0; }
      .sids-table-wrap {
        overflow: auto; max-height: 420px;
        border: 1px solid rgba(128, 128, 128, 0.25); border-radius: 10px;
      }
      .sids-table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
      .sids-table th, .sids-table td {
        padding: 6px 12px; text-align: left; white-space: nowrap;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
      }
      .sids-table th { background: rgba(128, 128, 128, 0.12); font-weight: 600; }
      .sids-bar-bg {
        display: inline-block; width: 90px; height: 8px; border-radius: 4px;
        background: rgba(128, 128, 128, 0.2); margin-right: 8px; vertical-align: middle;
      }
      .sids-bar { display: block; height: 8px; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

cached_get_baseline = st.cache_data(ttl=30)(api_client.get_baseline_metrics)
cached_get_baseline_findings = st.cache_data(ttl=30)(api_client.get_baseline_findings)
cached_get_comparison = st.cache_data(ttl=30)(api_client.get_comparison)
cached_get_evasion = st.cache_data(ttl=30)(api_client.get_evasion_rate)


@st.cache_data(ttl=60, show_spinner=False)
def _cached_features() -> dict:
    """Raises on failure so st.cache_data never caches an error (start the API -> works immediately)."""
    data, err = api_client.get_features()
    if err:
        raise RuntimeError(err)
    return data


def get_features():
    """(features, error) — features come from GET /defender/features."""
    try:
        return _cached_features(), None
    except RuntimeError as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def sidebar() -> str:
    st.sidebar.title("\U0001F6E1\uFE0F SecureIDS")

    features, err = get_features()
    if err is None:
        st.sidebar.success(f"API online · {features['model_used']}")
    else:
        st.sidebar.error("API offline or model not loaded")
        st.sidebar.caption(err)
    st.sidebar.caption(f"API: {api_client.API_BASE_URL}")

    if st.sidebar.button("\U0001F504 Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    page = st.sidebar.radio(
        "View",
        [
            "Overview",
            "Baseline Model Performance",
            "Evasion Rate (FGSM)",
            "Before / After Adversarial Training",
            "Live Prediction Demo",
        ],
    )
    return page


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def data_status_badge():
    _, baseline_err = cached_get_baseline()
    _, comparison_err = cached_get_comparison()
    _, evasion_err = cached_get_evasion()

    cols = st.columns(3)
    for col, label, err in zip(
        cols,
        ["Baseline metrics", "Comparison metrics", "Evasion metrics"],
        [baseline_err, comparison_err, evasion_err],
    ):
        with col:
            if err is None:
                st.success(f"{label}: ready")
            else:
                st.warning(f"{label}: not available yet")


def page_overview():
    st.title("SecureIDS Dashboard")
    st.write(
        "Network Intrusion Detection with adversarial attack simulation (FGSM) "
        "and adversarial training. This dashboard reads results produced by the "
        "Defender and Attacker pipelines through the API — it never computes "
        "metrics itself."
    )

    comparison, err = cached_get_comparison("all_eps")
    if not err:
        st.subheader("Headline results")
        baseline, robust = comparison["baseline"], comparison["robust"]
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Clean accuracy (robust model)",
            f"{robust['accuracy']:.1%}",
            delta=f"{robust['accuracy'] - baseline['accuracy']:+.1%} vs baseline",
        )
        c2.metric(
            "Adversarial detection rate",
            f"{robust['detection_rate']:.1%}",
            delta=f"{robust['detection_rate'] - baseline['detection_rate']:+.1%} vs baseline",
        )
        c3.metric(
            "Macro F1",
            f"{robust['macro_f1']:.1%}",
            delta=f"{robust['macro_f1'] - baseline['macro_f1']:+.1%} vs baseline",
        )

    st.subheader("Data status")
    data_status_badge()
    st.caption(
        "If something above says 'not available yet', that pipeline step "
        "hasn't been run, or the file hasn't landed in `results/metrics/` yet."
    )


# ---------------------------------------------------------------------------
# Metrics pages
# ---------------------------------------------------------------------------

def page_baseline():
    st.title("Baseline Model Performance")
    data, err = cached_get_baseline()
    if err:
        st.warning(err)
        st.info("Run `src/defender/train_baseline.py` to produce `results/metrics/baseline_metrics.json`.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", data["model_name"])
    col2.metric("Accuracy", f"{data['accuracy']:.1%}")
    col3.metric("Macro F1", f"{data['macro_f1']:.1%}")

    if data.get("notes"):
        st.info(data["notes"])

    left, right = st.columns(2)
    with left:
        st.plotly_chart(plots.per_class_metrics_bar(data["per_class"]), width="stretch")
    with right:
        st.plotly_chart(
            plots.confusion_matrix_heatmap(data["confusion_matrix"], data["classes"]),
            width="stretch",
        )

    findings, findings_err = cached_get_baseline_findings()
    if not findings_err:
        with st.expander("Read the Defender's full findings writeup"):
            st.markdown(findings["markdown"])


def page_evasion():
    st.title("FGSM Evasion Rate")
    data, err = cached_get_evasion()
    if err:
        st.warning(err)
        st.info("Run the attacker's evasion script to produce `results/metrics/evasion_metrics.json`.")
        return

    st.caption(f"Attack type: {data.get('attack_type', 'FGSM')} · Target model: {data.get('target_model', 'n/a')}")

    has_confidence_drop = any(r.get("confidence_drop") is not None for r in data["results"])
    metric = "evasion_rate"
    if has_confidence_drop:
        metric = st.radio(
            "Metric",
            ["evasion_rate", "confidence_drop"],
            horizontal=True,
            format_func=lambda m: m.replace("_", " ").capitalize(),
        )

    st.plotly_chart(plots.evasion_rate_by_epsilon(data["results"], metric=metric), width="stretch")
    st.caption(
        "Higher epsilon = larger perturbation budget for the attacker. "
        "A rising curve here is expected — the interesting question is how "
        "much adversarial training (next tab) flattens it."
    )


def page_comparison():
    st.title("Before / After Adversarial Training")

    experiment = st.radio(
        "Adversarial training run",
        ["all_eps", "high_eps_only"],
        format_func=lambda x: "Trained on all epsilons" if x == "all_eps" else "Trained on high-epsilon samples only",
        horizontal=True,
    )

    data, err = cached_get_comparison(experiment)
    if err:
        st.warning(err)
        st.info(
            f"Run `src/defender/adversarial_training.py` to produce "
            f"`results/metrics/comparison_{experiment}.json`."
        )
        return

    st.caption(f"Experiment: {data.get('experiment_name', experiment)}")

    baseline, robust = data["baseline"], data["robust"]
    st.plotly_chart(plots.before_after_comparison(baseline, robust), width="stretch")

    delta_acc = robust["accuracy"] - baseline["accuracy"]
    delta_det = robust["detection_rate"] - baseline["detection_rate"]
    col1, col2 = st.columns(2)
    col1.metric("Clean accuracy change", f"{robust['accuracy']:.1%}", delta=f"{delta_acc:+.1%}")
    col2.metric("Adversarial detection rate change", f"{robust['detection_rate']:.1%}", delta=f"{delta_det:+.1%}")

    if delta_det > 0 and delta_acc >= -0.02:
        st.success("Adversarial training improved robustness without a meaningful hit to clean accuracy.")
    elif delta_det > 0:
        st.info("Robustness improved, but at a noticeable cost to clean accuracy — worth discussing trade-offs.")
    else:
        st.warning("Detection rate didn't improve after adversarial training — worth investigating.")


# ---------------------------------------------------------------------------
# Live Prediction Demo
# ---------------------------------------------------------------------------

MAX_TABLE_ROWS = 500  # HTML tables get slow with thousands of rows; the CSV download has everything


def _show_table(headers: list, rows: list, raw_html_columns: tuple = ()) -> None:
    """
    Render a table as escaped HTML.

    Deliberately NOT st.dataframe / st.data_editor / st.table: those need
    pyarrow, and on machines where an Application Control policy blocks
    pyarrow's DLL they crash the whole page with an ImportError.
    Every cell is HTML-escaped (uploaded CSV content is untrusted) except
    columns listed in raw_html_columns, which this file builds itself.
    """
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{v if headers[i] in raw_html_columns else html.escape(str(v))}</td>"
            for i, v in enumerate(row)
        )
        body_rows.append(f"<tr>{cells}</tr>")
    st.html(
        f'<div class="sids-table-wrap"><table class="sids-table">'
        f'<thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'
    )


def _confidence_cell(conf: float, color: str) -> str:
    pct = max(0.0, min(1.0, conf)) * 100
    return (
        f'<span class="sids-bar-bg"><span class="sids-bar" style="width:{pct:.1f}%;background:{color}"></span></span>'
        f"{pct:.1f}%"
    )


# Headline flow features offered in the manual-entry form. Every other one of
# the 78 features is filled with its training-set average.
KEY_FEATURES = [
    "Dst Port", "Protocol", "Flow Duration",
    "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Flow Byts/s", "Flow Pkts/s", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Mean", "SYN Flag Cnt", "ACK Flag Cnt", "RST Flag Cnt", "Init Fwd Win Byts",
]


def _norm(name) -> str:
    return str(name).strip().lower()


def _template_csv(features: dict) -> bytes:
    cols = features["feature_columns"]
    row = [features["feature_means"][c] for c in cols]
    return pd.DataFrame([row], columns=cols).to_csv(index=False).encode()


def _classify(csv_bytes: bytes, source_name: str) -> None:
    """Call the API and park the outcome in session_state so it survives reruns."""
    with st.spinner("Classifying ..."):
        result, err = api_client.predict(csv_bytes, source_name)
    if err:
        st.session_state.pop("demo_result", None)
        st.session_state["demo_error"] = err
    else:
        st.session_state["demo_result"] = {"result": result, "source": source_name}
        st.session_state.pop("demo_error", None)


def _upload_tab(features: dict) -> None:
    uploaded = st.file_uploader("Traffic data (CSV)", type=["csv"], key="demo_uploader")
    if uploaded is None:
        st.info("Upload a CSV, or use the **Manual entry** tab to try it without a file.")
        return

    raw = uploaded.getvalue()
    try:
        preview_df = pd.read_csv(io.BytesIO(raw), nrows=10)
    except Exception as e:  # noqa: BLE001
        st.error(f"Couldn't parse this as CSV: {e}")
        return

    approx_rows = max(raw.count(b"\n") - 1, 0)
    st.caption(f"{uploaded.name} · about {approx_rows:,} rows · {len(preview_df.columns)} columns · preview of first 10 rows")
    _show_table(list(preview_df.columns), preview_df.fillna("").values.tolist())

    have = {_norm(c) for c in preview_df.columns}
    missing = [c for c in features["feature_columns"] if _norm(c) not in have]
    if missing:
        st.error(
            f"This file is missing {len(missing)} of the {len(features['feature_columns'])} "
            f"required feature columns (e.g. {', '.join(missing[:4])}). "
            "Download the template above to see the exact headers."
        )
    else:
        extra = "" if any(_norm(c) == "label" for c in preview_df.columns) else " (add a `Label` column to compare against the truth)"
        st.success(f"All {len(features['feature_columns'])} feature columns found{extra}.")

    if st.button("Classify", type="primary", disabled=bool(missing), key="classify_upload"):
        _classify(raw, uploaded.name)


def _manual_tab(features: dict) -> None:
    all_cols = features["feature_columns"]
    means = features["feature_means"]
    shown = [c for c in KEY_FEATURES if c in all_cols] or all_cols[:12]

    st.caption(
        "Describe one network flow using the headline features below. The other "
        f"{len(all_cols) - len(shown)} features are set to their training-set average. That average isn't a real "
        "network flow, so the model can be confidently wrong on it — results are only as "
        "meaningful as the traffic you describe. To classify many flows at once, use the CSV upload."
    )

    values = {}
    grid = st.columns(4)
    for i, c in enumerate(shown):
        with grid[i % 4]:
            values[c] = st.number_input(
                c, value=float(round(means[c], 2)), step=1.0, format="%.2f", key=f"manual_{c}"
            )

    if st.button("Classify this flow", type="primary", key="classify_manual"):
        full = dict(means)
        full.update(values)
        csv_bytes = pd.DataFrame([[full[c] for c in all_cols]], columns=all_cols).to_csv(index=False).encode()
        _classify(csv_bytes, "manual_entry.csv")


def _results_frame(result: dict) -> pd.DataFrame:
    preds = result["predictions"]
    has_actual = any(p.get("actual_label") for p in preds)
    df = pd.DataFrame(
        {
            "Row #": [p["row_index"] + 1 for p in preds],
            "Verdict": [
                "\U0001F7E2 Benign" if _norm(p["predicted_class"]) == "benign" else "\U0001F534 Attack"
                for p in preds
            ],
            "Predicted class": [p["predicted_class"] for p in preds],
            "Confidence": [p["confidence"] for p in preds],
        }
    )
    if has_actual:
        df["Actual label"] = [p.get("actual_label") or "" for p in preds]
        df["Match"] = [
            "\u2705" if _norm(p["predicted_class"]) == _norm(p.get("actual_label") or "") else "\u274C"
            for p in preds
        ]
    return df


def _render_results(result: dict, source: str) -> None:
    preds = result["predictions"]
    df = _results_frame(result)
    n_attack = int((df["Verdict"].str.contains("Attack")).sum())

    st.subheader("Results")
    st.caption(f"Source: {source} · model: {result['model_used']}")

    skipped = result.get("skipped_rows", [])
    if skipped:
        st.warning(
            f"{len(skipped)} of {result['n_rows']} row(s) were not classified because they contain "
            "missing, infinite or non-numeric values (the model was trained with such rows removed)."
        )
        with st.expander("Which rows were skipped?"):
            _show_table(
                ["Row #", "Reason"],
                [[x["row_index"] + 1, x["reason"]] for x in skipped[:MAX_TABLE_ROWS]],
            )

    cols = st.columns(4 if "Match" in df.columns else 3)
    cols[0].metric("Rows classified", f"{len(preds):,}")
    cols[1].metric("Flagged as attack", f"{n_attack:,}")
    cols[2].metric("Benign", f"{len(preds) - n_attack:,}")
    if "Match" in df.columns:
        agree = (df["Match"] == "\u2705").mean()
        cols[3].metric("Agrees with Label column", f"{agree:.1%}")

    if len(preds) > 1:
        counts = df["Predicted class"].value_counts().to_dict()
        st.plotly_chart(plots.prediction_summary_bar(counts), width="stretch")

    shown_df = df.head(MAX_TABLE_ROWS)
    show_headers = list(shown_df.columns)
    table_rows = []
    for _, r in shown_df.iterrows():
        is_attack = "Attack" in r["Verdict"]
        row = list(r.values)
        row[show_headers.index("Confidence")] = _confidence_cell(
            r["Confidence"], plots.ATTACK_COLOR if is_attack else plots.BENIGN_COLOR
        )
        table_rows.append(row)
    _show_table(show_headers, table_rows, raw_html_columns=("Confidence",))
    if len(df) > MAX_TABLE_ROWS:
        st.caption(f"Showing the first {MAX_TABLE_ROWS:,} of {len(df):,} rows — download the CSV for all of them.")

    # Per-row drill-down: the full probability distribution behind one verdict.
    by_row = {p["row_index"] + 1: p for p in preds}
    row_numbers = list(by_row)
    chosen = row_numbers[0] if len(row_numbers) == 1 else st.selectbox(
        "Inspect one row's full probability breakdown", row_numbers, key="inspect_row"
    )
    p = by_row[chosen]
    st.plotly_chart(plots.probability_bar(p["probabilities"], p["predicted_class"]), width="stretch")

    export = pd.DataFrame(
        [
            {"row": q["row_index"] + 1, "predicted_class": q["predicted_class"], "confidence": q["confidence"],
             **{f"p_{k}": v for k, v in q["probabilities"].items()}}
            for q in preds
        ]
    )
    left, right = st.columns([1, 1])
    left.download_button("Download results (CSV)", export.to_csv(index=False).encode(), "secureids_predictions.csv", "text/csv")
    if right.button("Clear results"):
        st.session_state.pop("demo_result", None)
        st.rerun()


def page_predict_demo():
    st.title("Live Prediction Demo")
    st.caption("Send traffic-flow records to the trained IDS and see how it classifies each one — live, via `POST /defender/predict`.")

    features, err = get_features()
    if err:
        st.error(err)
        st.info("Start the API (`python -m uvicorn api.main:app --reload` from the project root), then click **Refresh data** in the sidebar.")
        return

    n_features = len(features["feature_columns"])
    st.caption(
        f"Model: **{features['model_used']}** · {n_features} features · "
        f"classes: {', '.join(features['class_names'])}"
    )

    with st.expander("What file do I need?"):
        st.markdown(
            f"- A CSV with one row per network flow and **all {n_features} feature columns** (raw, unscaled values — the API scales them).\n"
            "- Header spacing and capitalisation don't matter (`Flow Duration` = ` flow duration`).\n"
            "- Extra columns are ignored. If there's a `Label` column, the results are compared against it.\n"
            "- Rows with blank, `NaN` or `Infinity` values can't be classified and are reported as skipped."
        )
        st.download_button(
            "Download CSV template (one row of training averages)",
            _template_csv(features),
            "secureids_template.csv",
            "text/csv",
        )

    tab_upload, tab_manual = st.tabs(["Upload CSV", "Manual entry"])
    with tab_upload:
        _upload_tab(features)
    with tab_manual:
        _manual_tab(features)

    if st.session_state.get("demo_error"):
        st.error(st.session_state["demo_error"])
    saved = st.session_state.get("demo_result")
    if saved:
        st.divider()
        _render_results(saved["result"], saved["source"])


PAGES = {
    "Overview": page_overview,
    "Baseline Model Performance": page_baseline,
    "Evasion Rate (FGSM)": page_evasion,
    "Before / After Adversarial Training": page_comparison,
    "Live Prediction Demo": page_predict_demo,
}


def main():
    selected_page = sidebar()
    PAGES[selected_page]()


if __name__ == "__main__":
    main()