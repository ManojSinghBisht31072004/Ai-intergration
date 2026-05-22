import streamlit as st
import requests

API = "http://127.0.0.1:8000"

st.title("Knowledge Base Assistant")

# ── Upload ─────────────────────────────────────────────────────────────────
st.subheader("Upload a document")
file = st.file_uploader("Choose PDF or TXT", type=["pdf", "txt", "pdf"])

if file:
    # auto-ingest as soon as file is selected — no button needed
    if st.session_state.get("last_uploaded") != file.name:
        with st.spinner(f"Processing '{file.name}'..."):
            res = requests.post(
                f"{API}/upload",
                files={"file": (file.name, file.getvalue(), file.type)}
            )
        if res.status_code == 200:
            data = res.json()
            st.session_state.last_uploaded = file.name
            st.success(f"Ready! {data['total_chunks']} chunks stored from '{data['filename']}'")
        else:
            st.error(f"Upload failed: {res.text}")
    else:
        st.success(f"'{file.name}' already ingested and ready.")

st.divider()

# ── Ask ────────────────────────────────────────────────────────────────────
st.subheader("Ask a question")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Your question")

if st.button("Ask") and question.strip():
    with st.spinner("Searching documents..."):
        res = requests.post(
            f"{API}/ask",
            json={"question": question, "top_k": 5}
        )
    if res.status_code == 200:
        data = res.json()
        st.session_state.history.append({"q": question, "a": data})
    else:
        st.error(f"Error: {res.text}")

# ── History ────────────────────────────────────────────────────────────────
for item in reversed(st.session_state.history):
    st.markdown(f"**Q:** {item['q']}")
    st.markdown(f"**A:** {item['a']['answer']}")

    if item['a']['sources']:
        with st.expander("View sources"):
            for src in item['a']['sources']:
                st.caption(f"`{src['chunk_id']}` — {src['snippet']}")

    if item['a']['hallucination_detected']:
        st.warning("Hallucination detected in this response.")

    st.divider()