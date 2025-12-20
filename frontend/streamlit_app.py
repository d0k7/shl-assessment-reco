import os
import requests
import streamlit as st
import pandas as pd

st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

DEFAULT_API = os.getenv("API_BASE", "https://shl-assessment-reco-t8zx.onrender.com").rstrip("/")

st.title("SHL Assessment Recommender")
st.caption("Enter a natural language query, paste a JD, or paste a JD URL. Returns SHL assessment recommendations.")

api_base = st.text_input("API Base", value=DEFAULT_API).rstrip("/")

query = st.text_area(
    "Query / JD Text / JD URL",
    value="Need a Java developer who is good in collaborating with external teams and stakeholders.",
    height=180,
)

top_k = st.slider("Top K", min_value=5, max_value=10, value=10)

if st.button("Recommend"):
    if not query.strip():
        st.error("Query cannot be empty.")
        st.stop()

    try:
        health = requests.get(f"{api_base}/health", timeout=20)
        health.raise_for_status()
    except Exception as e:
        st.error(f"API health check failed: {e}")
        st.stop()

    payload = {"query": query, "top_k": top_k}

    try:
        r = requests.post(f"{api_base}/recommend", json=payload, timeout=60)
        if r.status_code >= 400:
            # show backend error detail if FastAPI returned it
            st.error(f"/recommend failed: {r.status_code}")
            try:
                st.json(r.json())
            except Exception:
                st.text(r.text)
            st.stop()
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

    # Make URL clickable if present
    if "url" in df.columns:
        df["url"] = df["url"].apply(lambda u: f"[link]({u})" if isinstance(u, str) and u else "")

    # Ensure score visible
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce").round(6)

    st.subheader("Recommendations")
    st.dataframe(df, use_container_width=True)

    with st.expander("Raw JSON"):
        st.json(data)
