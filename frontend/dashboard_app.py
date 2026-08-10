"""
SecureIDS Dashboard — Visualizer's main Streamlit app.

Run with:
    streamlit run frontend/dashboard_app.py

Only talks to the FastAPI backend via api_client.py — never reads
results/metrics/*.json directly.
"""

import streamlit as st

import api_client
import plots

st.set_page_config(page_title="SecureIDS Dashboard", page_icon="\U0001F6E1\uFE0F", layout="wide")

cached_get_baseline = st.cache_data(ttl=30)(api_client.get_baseline_metrics)
cached_get_baseline_findings = st.cache_data(ttl=30)(api_client.get_baseline_findings)
cached_get_comparison = st.cache_data(ttl=30)(api_client.get_comparison)
cached_get_evasion = st.cache_data(ttl=30)(api_client.get_evasion_rate)


def sidebar() -> str:
    st.sidebar.title("\U0001F6E1\uFE0F SecureIDS")
    st.sidebar.caption(f"API: {api_client.API_BASE_URL}")

    if st.sidebar.button("\U0001F504 Refresh data"):
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
    st.subheader("Data status")
    data_status_badge()
    st.caption(
        "If something above says 'not available yet', that pipeline step "
        "hasn't been run, or the file hasn't landed in `results/metrics/` yet."
    )


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
        st.plotly_chart(plots.per_class_metrics_bar(data["per_class"]), use_container_width=True)
    with right:
        st.plotly_chart(
            plots.confusion_matrix_heatmap(data["confusion_matrix"], data["classes"]),
            use_container_width=True,
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
            "Metric", ["evasion_rate", "confidence_drop"], horizontal=True, format_func=str.title
        )

    st.plotly_chart(plots.evasion_rate_by_epsilon(data["results"], metric=metric), use_container_width=True)
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
    st.plotly_chart(plots.before_after_comparison(baseline, robust), use_container_width=True)

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


def page_predict_demo():
    st.title("Live Prediction Demo")
    st.caption(
        "Stretch goal from the original brief: upload traffic data and trigger "
        "a live classification against the trained IDS via `POST /defender/predict`."
    )

    baseline_data, _ = cached_get_baseline()
    if baseline_data:
        st.caption(f"Expected classes: {', '.join(baseline_data['classes'])}")

    uploaded = st.file_uploader("Traffic data (CSV)", type=["csv"])
    if uploaded is None:
        st.info("Upload a file to preview it and try a live prediction.")
        return

    try:
        import pandas as pd

        preview_df = pd.read_csv(uploaded)
        st.write(f"{len(preview_df)} rows · {len(preview_df.columns)} columns")
        st.dataframe(preview_df.head(10), use_container_width=True)
        uploaded.seek(0)
    except Exception as e:
        st.warning(f"Couldn't parse this as CSV: {e}")
        return

    if st.button("Classify"):
        with st.spinner("Calling /defender/predict ..."):
            result, err = api_client.predict(uploaded.getvalue(), uploaded.name)
        if err:
            st.warning(err)
            st.caption(
                "This endpoint is owned by the Defender (`api/routers/defender_routes.py`). "
                "The dashboard is wired up and will work as soon as it's implemented — "
                "nothing else needs to change here."
            )
        else:
            st.success("Prediction complete.")
            st.json(result)


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