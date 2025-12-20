import os
import requests
import streamlit as st
import pandas as pd

st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")

st.title("SHL Assessment Recommender")
st.caption("Enter a natural language query, paste a JD, or paste a JD URL. Returns SHL assessment recommendations.")

query = st.text_area(
    "Query / JD Text / JD URL",
    value="Need a Java developer who is good in collaborating with external teams and stakeholders.",
    height=180,
)

top_k = st.slider("Top K", min_value=5, max_value=10, value=10)

st.write("**API Base**")
st.code(API_BASE)

if st.button("Recommend"):
    if not query.strip():
        st.error("Query cannot be empty.")
        st.stop()

    try:
        health = requests.get(f"{API_BASE}/health", timeout=15)
        health.raise_for_status()
    except Exception as e:
        st.error(f"API health check failed: {e}")
        st.stop()

    payload = {"query": query, "top_k": top_k}
    try:
        r = requests.post(f"{API_BASE}/recommend", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        st.error(f"/recommend failed: {e}")
        st.stop()

    recs = data.get("recommended_assessments", [])
    if not recs:
        st.warning("No recommendations returned.")
        st.json(data)
        st.stop()

    df = pd.DataFrame(recs)

    # Make sure these columns exist (robust)
    for col in ["description", "duration", "remote_support", "adaptive_support", "test_type", "score"]:
        if col not in df.columns:
            df[col] = None

    # Render list nicely
    df["test_type"] = df["test_type"].apply(lambda x: ", ".join(x) if isinstance(x, list) else (x or ""))

    # Order columns
    cols = ["name", "url", "score", "duration", "remote_support", "adaptive_support", "test_type", "description"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    st.subheader("Recommendations")
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "url": st.column_config.LinkColumn("URL"),
        },
    )

    with st.expander("Raw JSON"):
        st.json(data)
