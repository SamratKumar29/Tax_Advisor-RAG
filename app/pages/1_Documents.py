"""
Documents page: add your own books / notes to the knowledge base, or remove them.

UI only — validation, chunking, embedding and storage live in src.ingest.user_docs.
"""

import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

from src import config
from src.ingest import user_docs

st.title("📄 Your documents")
st.caption(
    "Add a PDF, Markdown or text file (for example a tax book or your own notes). It is "
    "split into chunks, embedded locally and searched together with the Act and Rules, "
    "tagged as a *supplementary* source. Files stay on this machine under "
    f"`{Path(config.USER_DOCS_DIR).name}/` and are never committed to git."
)
st.info(
    "Only upload documents you are allowed to use. Record the licence or permission below; "
    "it is stored with the document."
)

with st.form("upload", clear_on_submit=True):
    uploaded = st.file_uploader(
        f"File (max {config.MAX_UPLOAD_MB} MB, {config.MAX_UPLOAD_PAGES} pages)",
        type=sorted(s.lstrip(".") for s in user_docs.ALLOWED_SUFFIXES),
    )
    licence = st.text_input("Licence / permission",
                            placeholder="e.g. CC-BY 4.0, own notes, publisher licence")
    submitted = st.form_submit_button("Add to knowledge base")

if submitted:
    if uploaded is None:
        st.error("Choose a file first.")
    else:
        try:
            with st.spinner("Chunking and embedding..."):
                entry = user_docs.add_document(uploaded.name, uploaded.getvalue(), licence)
        except ValueError as e:
            st.error(f"⚠️ {e}")
        else:
            st.success(f"✅ Added `{entry['filename']}` ({entry['n_chunks']} chunks).")

st.subheader("Uploaded documents")
docs = user_docs.list_documents()
if not docs:
    st.caption("None yet.")
for doc in docs:
    cols = st.columns([4, 1])
    with cols[0]:
        st.markdown(f"**{doc['filename']}** · {doc['n_chunks']} chunks · licence: {doc['licence']}")
        st.caption(f"Added {doc['uploaded_at']} · sha256 {doc['sha256'][:12]}…")
    with cols[1]:
        if st.button("Delete", key=f"delete_{doc['source_doc']}"):
            removed = user_docs.delete_document(doc["source_doc"])
            st.success(f"Removed {doc['filename']} ({removed} chunks).")
            st.rerun()
