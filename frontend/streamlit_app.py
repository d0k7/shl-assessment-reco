import os
import requests
import streamlit as st
import pandas as pd

st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

DEFAULT_API_BASE = "https://shl-assessment-reco-t8zx.onrender.com"
API_BASE = os.getenv("API_BASE", DEFAULT_API_BASE).rstrip("/")

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
        health = requests.get(f"{API_BASE}/health", timeout=10)
        health.raise_for_status()
    except Exception as e:
        st.error(f"API health check failed: {e}")
        st.stop()

    payload = {"query": query, "top_k": top_k}
    try:
        r = requests.post(f"{API_BASE}/recommend", json=payload, timeout=30)
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
    df["url"] = df["url"].apply(lambda u: f"[link]({u})")

    st.subheader("Recommendations")
    st.dataframe(df, use_container_width=True)

    with st.expander("Raw JSON"):
        st.json(data)
