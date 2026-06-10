"""
app.py — Flickr30k Multimodal RAG Chatbot
  • Query is embedded by CLIP (same space as images + captions)
  • Single collection query returns top-1 image + top-4 captions
  • Retrieved results shown to user; user can then chat with LLM about them
"""

import os
import warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*__path__.*")

import streamlit as st
from PIL import Image

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
COLLECTION  = "flickr30k_subset"
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Flickr30k RAG Chatbot",
    page_icon="🖼️",
    layout="centered",
)

# ── ChromaDB ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = OpenCLIPEmbeddingFunction()
    il = ImageLoader()
    try:
        return client.get_collection(
            name=COLLECTION,
            embedding_function=ef,
            data_loader=il,
        )
    except Exception:
        return None

collection = get_collection()

# ── LLM ───────────────────────────────────────────────────────────────────────
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

SYSTEM_PROMPT = """You are a helpful visual assistant.
You have been given a retrieved image and related captions from a photo database.
Use the captions as your description of the image to answer the user's questions.
If the user asks something unrelated, answer naturally."""

@st.cache_resource
def get_llm(repo, token):
    llm = HuggingFaceEndpoint(
        repo_id=repo,
        huggingfacehub_api_token=token,
        max_new_tokens=512,
    )
    return ChatHuggingFace(llm=llm)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    repo_id = st.text_input(
        "HF Model Repo",
        value="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    st.divider()
    st.markdown("**Vector DB**")
    if collection:
        st.success(f"Connected\n`{COLLECTION}`\n{collection.count()} items")
    else:
        st.error("Collection not found.\nRun `python ingest.py` first.")
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        for k in ["messages", "lc_history", "retrieved_image",
                  "retrieved_captions", "search_done"]:
            st.session_state.pop(k, None)
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "messages"          : [],
    "lc_history"        : [SystemMessage(content=SYSTEM_PROMPT)],
    "retrieved_image"   : None,
    "retrieved_captions": [],
    "search_done"       : False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Guards ────────────────────────────────────────────────────────────────────
st.title("Flickr30k RAG Chatbot")
st.caption("Search the photo database — retrieve the best matching image and chat about it.")

if not hf_token:
    st.error("Add `HUGGINGFACEHUB_API_TOKEN` to your `.env` file.")
    st.stop()
if not collection:
    st.error("Run `python ingest.py` to build the vector database first.")
    st.stop()

try:
    chat_model = get_llm(repo_id, hf_token)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str):
    """
    Single query against the shared CLIP vector space.
    Returns:
        image_path  : str | None
        image_name  : str | None
        captions    : list[str]  — top-4 caption strings
    """
    # ── top-1 image ──────────────────────────────────────────────────────────
    # Query the collection filtering to image-type entries only.
    img_res = collection.query(
        query_texts=[query],
        n_results=1,
        where={"type": "image"},
        include=["metadatas", "uris", "distances"],
    )

    image_path = None
    image_name = None
    uris_list  = img_res.get("uris") or []
    metas_list = img_res.get("metadatas") or []

    if uris_list and uris_list[0]:
        image_path = uris_list[0][0]
    if metas_list and metas_list[0]:
        image_name = metas_list[0][0].get("image_name", "")

    # ── top-4 captions ───────────────────────────────────────────────────────
    # Same vector space: CLIP-encoded query text finds nearest caption vectors.
    cap_res = collection.query(
        query_texts=[query],
        n_results=4,
        where={"type": "text"},
        include=["documents", "metadatas", "distances"],
    )

    captions   = []
    docs_list  = cap_res.get("documents") or []
    dists_list = cap_res.get("distances") or []

    if docs_list and docs_list[0]:
        for doc, dist in zip(docs_list[0], dists_list[0] if dists_list else []):
            if doc:
                captions.append({"text": doc, "distance": round(dist, 3)})

    return image_path, image_name, captions

# ── Chat history display ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        for item in msg["content"]:
            if isinstance(item, str):
                st.markdown(item)
            elif isinstance(item, Image.Image):
                st.image(item, use_container_width=True)

# ── Phase 1: Search bar ───────────────────────────────────────────────────────
if not st.session_state.search_done:
    st.markdown("#### Search the photo database")
    col1, col2 = st.columns([4, 1])
    with col1:
        search_query = st.text_input(
            "query",
            placeholder="e.g. a dog playing in the snow",
            label_visibility="collapsed",
        )
    with col2:
        search_clicked = st.button("Search", use_container_width=True, type="primary")

    if search_clicked and search_query.strip():
        with st.spinner("Searching vector database..."):
            img_path, img_name, captions = retrieve(search_query.strip())

        if img_path and os.path.exists(img_path):
            pil_img = Image.open(img_path).convert("RGB")
            st.session_state.retrieved_image    = pil_img
            st.session_state.retrieved_captions = captions
            st.session_state.search_done        = True

            # Build context block for the LLM (injected once, silently)
            cap_lines = "\n".join(
                [f"  {i+1}. {c['text']}  [similarity distance: {c['distance']}]"
                 for i, c in enumerate(captions)]
            )
            context_msg = (
                f'User searched for: "{search_query}"\n\n'
                f"Most relevant image: {img_name}\n\n"
                f"Top 4 related captions from the database:\n{cap_lines}\n\n"
                "These captions describe what is in the image. "
                "Use them to answer the user's follow-up questions."
            )
            st.session_state.lc_history.append(HumanMessage(content=context_msg))
            st.session_state.lc_history.append(
                AIMessage(content=(
                    "Understood. I can see the retrieved image and its captions. "
                    "I'm ready to answer questions about it."
                ))
            )
            st.rerun()
        else:
            st.warning("No matching image found. Try a different query.")

# ── Phase 2: Results panel + chat ────────────────────────────────────────────
if st.session_state.search_done:

    # Results panel
    with st.container(border=True):
        img_col, cap_col = st.columns([1, 1])

        with img_col:
            st.markdown("**Top 1 Retrieved Image**")
            if st.session_state.retrieved_image:
                st.image(st.session_state.retrieved_image, use_container_width=True)

        with cap_col:
            st.markdown("**Top 4 Related Captions**")
            for i, cap in enumerate(st.session_state.retrieved_captions, 1):
                dist  = cap.get("distance", "")
                text  = cap.get("text", "")
                label = f"distance: {dist}" if dist != "" else ""
                st.markdown(f"**{i}.** {text}")
                if label:
                    st.caption(label)

    st.divider()
    st.markdown("#### Chat about this image")

    # Chat input
    prompt = st.chat_input("Ask anything about the retrieved image...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": [prompt]})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    st.session_state.lc_history.append(HumanMessage(content=prompt))
                    result     = chat_model.invoke(st.session_state.lc_history)
                    reply_text = result.content
                    st.markdown(reply_text)
                    st.session_state.lc_history.append(AIMessage(content=reply_text))
                    st.session_state.messages.append(
                        {"role": "assistant", "content": [reply_text]}
                    )
                except Exception as e:
                    err = f"Error: {e}"
                    st.error(err)