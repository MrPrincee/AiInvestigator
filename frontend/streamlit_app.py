import os

import requests
import streamlit as st


try:
    API_URL = st.secrets["API_URL"]
except (KeyError, FileNotFoundError):
    API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

API_URL = API_URL.rstrip("/")

def api_request(method, path, **kwargs):
    response = requests.request(
        method,
        f"{API_URL}{path}",
        timeout=kwargs.pop("timeout", 60),
        **kwargs,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API error ({response.status_code}): {detail}") from error
    if not response.content:
        return {}
    return response.json()


def show_error(error):
    st.error(str(error))


def check_backend_status():
    """Check backend availability without calling Gemini or changing documents."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=(5, 15))
        response.raise_for_status()
        data = response.json()
        return "connected" if data.get("status") == "ok" else "unavailable"
    except requests.Timeout:
        return "waking_up"
    except (requests.RequestException, ValueError):
        return "unavailable"


def investigation_page():
    st.header("AI Investigation")
    st.caption("Ask questions across documents and inspect supporting evidence.")

    with st.sidebar:
        st.subheader("Investigation Settings")
        retrieval_mode = st.selectbox(
            "Retrieval Mode",
            ["standard", "tool_calling"],
            format_func=lambda value: "Standard RAG" if value == "standard" else "Tool Calling",
        )
        top_k = st.slider("Top K", 1, 20, 8)
        max_hops = st.slider("Maximum Search Hops", 1, 3, 3)

    with st.form("investigation_form"):
        question = st.text_area(
            "Investigation Question",
            placeholder="Which documents support the answer to this question?",
            height=120,
        )
        submitted = st.form_submit_button(
            "Investigate", type="primary", use_container_width=True
        )

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Investigating documents..."):
                try:
                    st.session_state["investigation_result"] = api_request(
                        "POST",
                        "/rag/ask",
                        json={
                            "question": question.strip(),
                            "top_k": top_k,
                            "max_hops": max_hops,
                            "retrieval_mode": retrieval_mode,
                        },
                        timeout=240,
                    )
                except (requests.RequestException, RuntimeError, ValueError) as error:
                    st.session_state.pop("investigation_result", None)
                    show_error(error)

    result = st.session_state.get("investigation_result")
    if result is None:
        return

    st.divider()
    st.subheader("Investigation Result")
    if result.get("is_answerable"):
        st.success("Answer Found")
    else:
        st.warning("Insufficient Evidence")

    st.markdown(result.get("answer") or "No answer available.")

    if result.get("has_contradictions"):
        st.error("Contradictory Information Detected")
        for contradiction in result.get("contradictions", []):
            if isinstance(contradiction, dict):
                st.write(contradiction.get("description", ""))
                st.caption(f"Evidence IDs: {contradiction.get('source_ids', [])}")
            else:
                st.write(contradiction)

    sources = result.get("sources", [])
    st.divider()
    st.subheader(f"Evidence & Citations ({len(sources)})")
    if not sources:
        st.info("No cited sources available.")
    for source in sources:
        citation_id = source.get("citation_id", "?")
        title = source.get("document_title", "Unknown Document")
        with st.expander(f"[{citation_id}] {title}"):
            st.write(source.get("evidence", ""))
            st.caption(f"Document ID: {source.get('document_id', 'N/A')}")
            st.caption(f"Chunk ID: {source.get('chunk_id', 'N/A')}")
            if source.get("page_number") is not None:
                st.caption(f"Page: {source['page_number']}")

    history = result.get("search_history", [])
    if history:
        st.divider()
        st.subheader("Search History")
        for item in history:
            with st.expander(f"Hop {item.get('hop')} — {item.get('query')}"):
                st.write(f"Results found: {item.get('results_count', 0)}")


def upload_tab():
    st.subheader("Upload Documents")
    st.caption("Supported formats: PDF, JSON, JSONL, CSV")
    with st.form("upload_form", clear_on_submit=True):
        file = st.file_uploader(
            "Choose a document",
            type=["pdf", "json", "jsonl", "csv"],
        )
        strategy = st.selectbox(
            "Chunking Strategy",
            ["fixed_size", "sentence_aware"],
            key="upload_strategy",
        )
        submitted = st.form_submit_button("Upload & Index", type="primary")

    if submitted:
        if file is None:
            st.warning("Choose a file first.")
            return
        with st.spinner("Uploading and indexing document..."):
            try:
                result = api_request(
                    "POST",
                    "/documents/upload",
                    files={"file": (file.name, file.getvalue(), "application/octet-stream")},
                    params={"chunking_strategy": strategy},
                    timeout=300,
                )
                st.success(
                    f"Indexed: {result.get('indexed_count', 0)} | "
                    f"Duplicates: {result.get('duplicate_count', 0)}"
                )
                with st.expander("Upload details"):
                    st.json(result)
            except (requests.RequestException, RuntimeError, ValueError) as error:
                show_error(error)


def add_tab():
    st.subheader("Add a Text Document")
    with st.form("create_document_form", clear_on_submit=True):
        title = st.text_input("Document Title")
        content = st.text_area("Document Content", height=220)
        source = st.text_input("Source", value="user")
        strategy = st.selectbox(
            "Chunking Strategy",
            ["fixed_size", "sentence_aware"],
            key="create_strategy",
        )
        submitted = st.form_submit_button("Add Document", type="primary")

    if submitted:
        if not title.strip() or not content.strip():
            st.warning("Title and content are required.")
            return
        with st.spinner("Indexing document..."):
            try:
                result = api_request(
                    "POST",
                    "/documents",
                    json={
                        "title": title.strip(),
                        "content": content.strip(),
                        "source": source.strip() or "user",
                        "metadata": {},
                        "chunking_strategy": strategy,
                    },
                    timeout=300,
                )
                st.success("Document created successfully.")
                st.code(result.get("document_id", ""))
                st.caption(f"Chunks indexed: {result.get('chunks_affected', 0)}")
            except (requests.RequestException, RuntimeError, ValueError) as error:
                show_error(error)


def list_tab():
    st.subheader("Indexed Documents")
    if st.button("Refresh Document List", key="refresh_documents"):
        st.session_state.pop("document_list", None)

    if "document_list" not in st.session_state:
        with st.spinner("Loading documents..."):
            try:
                st.session_state["document_list"] = api_request("GET", "/documents")
            except (requests.RequestException, RuntimeError, ValueError) as error:
                show_error(error)
                return

    data = st.session_state["document_list"]
    documents = data.get("documents", [])
    st.metric("Total Documents", data.get("total", len(documents)))
    if not documents:
        st.info("No documents found.")
        return

    search = st.text_input("Filter by title or document ID")
    if search.strip():
        needle = search.casefold().strip()
        documents = [
            document
            for document in documents
            if needle in str(document.get("title", "")).casefold()
            or needle in str(document.get("document_id", "")).casefold()
        ]

    st.dataframe(
        [
            {
                "Title": item.get("title", ""),
                "Document ID": item.get("document_id", ""),
                "Source": item.get("source", ""),
                "Chunks": item.get("chunk_count", 0),
            }
            for item in documents
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.caption("Select a document ID below to inspect its complete content.")
    options = {item["document_id"]: item for item in documents}
    selected_id = st.selectbox(
        "Inspect Document",
        options=list(options),
        format_func=lambda doc_id: f"{options[doc_id].get('title', '')} — {doc_id[:12]}",
    )
    if st.button("Load Document", key="inspect_document"):
        try:
            document = api_request("GET", f"/documents/{selected_id}")
            st.session_state["inspected_document"] = document
        except (requests.RequestException, RuntimeError, ValueError) as error:
            show_error(error)

    inspected = st.session_state.get("inspected_document")
    if inspected and inspected.get("document_id") == selected_id:
        st.text_input("Document Title", value=inspected.get("title", ""), disabled=True)
        st.text_area("Document Content", value=inspected.get("content", ""), height=220, disabled=True)
        st.json(inspected.get("metadata", {}))


def edit_tab():
    st.subheader("Update a Document")
    document_id = st.text_input("Document ID", key="edit_document_id")
    if st.button("Load for Editing"):
        if not document_id.strip():
            st.warning("Enter a document ID.")
        else:
            try:
                st.session_state["edit_document"] = api_request(
                    "GET", f"/documents/{document_id.strip()}"
                )
            except (requests.RequestException, RuntimeError, ValueError) as error:
                st.session_state.pop("edit_document", None)
                show_error(error)

    document = st.session_state.get("edit_document")
    if not document:
        return
    st.caption(f"Editing document: {document['document_id']}")
    with st.form("edit_document_form"):
        title = st.text_input("Title", value=document.get("title", ""))
        content = st.text_area("Content", value=document.get("content", ""), height=250)
        strategy = st.selectbox(
            "Chunking Strategy", ["fixed_size", "sentence_aware"], key="edit_strategy"
        )
        submitted = st.form_submit_button("Save Changes", type="primary")

    if submitted:
        if not title.strip() or not content.strip():
            st.warning("Title and content are required.")
            return
        with st.spinner("Updating document..."):
            try:
                result = api_request(
                    "PUT",
                    f"/documents/{document['document_id']}",
                    json={
                        "title": title.strip(),
                        "content": content.strip(),
                        "metadata": document.get("metadata", {}),
                        "chunking_strategy": strategy,
                    },
                    timeout=300,
                )
                st.success(f"Document updated. Chunks affected: {result.get('chunks_affected', 0)}")
                st.session_state.pop("edit_document", None)
                st.session_state.pop("document_list", None)
            except (requests.RequestException, RuntimeError, ValueError) as error:
                show_error(error)


def delete_tab():
    st.subheader("Delete a Document")
    st.warning("Deleting a document also removes its indexed chunks from Qdrant.")
    document_id = st.text_input("Document ID to Delete", key="delete_document_id")
    confirm = st.checkbox("I understand this document will be deleted.")
    if st.button("Delete Document", type="primary", disabled=not confirm):
        if not document_id.strip():
            st.warning("Enter a document ID.")
            return
        try:
            result = api_request("DELETE", f"/documents/{document_id.strip()}", timeout=120)
            st.success(f"Document deleted. Chunks removed: {result.get('chunks_affected', 0)}")
            st.session_state.pop("document_list", None)
            st.session_state.pop("inspected_document", None)
            st.session_state.pop("edit_document", None)
        except (requests.RequestException, RuntimeError, ValueError) as error:
            show_error(error)


def documents_page():
    st.header("Document Management")
    st.caption("Upload, browse, create, update, and delete indexed documents.")
    upload, browse, add, edit, delete = st.tabs(
        ["Upload", "Browse", "Add", "Update", "Delete"]
    )
    with upload:
        upload_tab()
    with browse:
        list_tab()
    with add:
        add_tab()
    with edit:
        edit_tab()
    with delete:
        delete_tab()


st.title("🔎 AI Investigator")
st.caption("Evidence-based Multi-Document Investigation Engine")

with st.sidebar:
    page = st.radio("Navigation", ["AI Investigation", "Document Management"])
    st.divider()

    if st.button("Retry Connection", use_container_width=True):
        st.session_state.pop("backend_status", None)

    if "backend_status" not in st.session_state:
        with st.spinner("Checking backend connection..."):
            st.session_state["backend_status"] = check_backend_status()

    backend_status = st.session_state["backend_status"]
    if backend_status == "connected":
        st.success("Backend Connected")
    elif backend_status == "waking_up":
        st.warning("Backend may be waking up.")
        st.caption("Please wait a moment and click Retry Connection.")
    else:
        st.error("Backend Unavailable")
        st.caption("Check the backend URL or try again later.")

    st.divider()

if page == "AI Investigation":
    investigation_page()
else:
    documents_page()
