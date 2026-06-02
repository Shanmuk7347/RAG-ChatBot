import streamlit as st
from rag_chain import get_chain, get_retriever, delete_chat_collection, get_vectorstore
from ingest import build_vector_data
import os
import uuid
import shutil

# Setup app for RAG-Chatbot
st.set_page_config(
    page_title="RAG-Chatbot",
    page_icon="🤖",
    layout="centered"
)

def new_chat(name="New Chat"):
    return {
        "name": name,
        "id": uuid.uuid4().hex[:12],
        "docs": [],
        "messages": []
    }

st.title("🔍 Ask Your PDFs")
st.caption("Powered by Llama")
st.markdown("---")

# setting up session state to store previous conversations so that they wont disapper due to rerunning
if "chats" not in st.session_state:
    st.session_state.chats = [new_chat("FirstChat")]

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = st.session_state.chats[0]["id"]

def active_chat():
    for c in st.session_state.chats:
        if c["id"] == st.session_state.active_chat_id:
            return c


# Side bar for PDF uploading and List of uploaded PDFs
with st.sidebar:
    st.header("RAG-Chatbot")
    st.subheader("Chats")

    if st.button("➕ New Chat"):
        name = st.text_input("Chat Name")
        st.session_state.chats.append(new_chat(name if name else f"Chat {len(st.session_state.chats)+1}"))
        st.session_state.active_chat_id = st.session_state.chats[-1]["id"]
        st.rerun()

    for chat in st.session_state.chats:
        label = f"🔍{chat["name"]}" if chat["id"] == st.session_state.active_chat_id else f"{chat["name"]}"
        if st.button(label, use_container_width=True):
            st.session_state.active_chat_id = chat["id"]
            st.rerun()
    
    chat = active_chat()
    st.markdown("---")

    st.subheader("📂Ingested Documents")

    if chat["docs"]:
        for name in chat["docs"]:
            st.markdown(f"✅ {name}")

    else:
        st.markdown("No docs uploaded")

    st.markdown("---")

    uploaded_files = st.file_uploader(label="Upload Files", max_upload_size=20, accept_multiple_files=True, type="pdf")

    if uploaded_files:
        if st.button("Ingest and Vectorize"):
            os.makedirs(f"./Docs/{chat["id"]}", exist_ok = True)
            new_file_paths = []
            for file in uploaded_files:
                file_path = os.path.join(f"./Docs/{chat["id"]}", file.name)

                if file.name in chat["docs"]:
                    st.warning(f"{file.name} already exists, Skipping...")
                    continue
               
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                chat["docs"].append(file.name)
                new_file_paths.append(file_path) 

            if new_file_paths:
                with st.spinner("Ingesting files..."):
                    build_vector_data(file_paths=new_file_paths, chat_id=chat["id"])
                    get_vectorstore.clear()
                st.success("Ingestion complete!")
                st.rerun()
            else:
                st.info("No new files for ingestion.")

    st.markdown("---")

    if st.button("🚮Delete This chat"):
        if os.path.exists(f"./Docs/{chat["id"]}"):
            shutil.rmtree(f"./Docs/{chat["id"]}")
        delete_chat_collection(chat["id"])
        get_vectorstore.clear()
        st.session_state.chats.remove(chat)
        if not st.session_state.chats:
            new = new_chat("New Chat")
            st.session_state.chats.append(new)
        st.session_state.active_chat_id = st.session_state.chats[0]["id"]
        st.rerun()
        
for message in chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.caption(source["file"])
                    st.markdown(source["preview"])

# User input    
if prompt := st.chat_input("Ask anything"):
    with st.chat_message("user"):
        st.markdown(prompt)
    # Storing user query in session state to display even after rerunning
    chat["messages"].append({"role": "user", "content": prompt})

# Invoke retriever and chain to get the answer and sources
    with st.spinner("Searching through the docs..."):
        st.write(
    "Current Chat:",
    chat["name"],
    chat["id"],
    "Docs:",
    len(chat["docs"])
)
        sources = get_retriever(chat["id"]).invoke(prompt)
        answer = get_chain(chat["id"]).invoke(prompt)
# Retriver gives relavent documents along with the top 4 chunks that LLM uses as Context, whereas Chain gives the answer given by LLM

# Displaying the answer and sources
    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("Sources"):
            for doc in sources:
                st.caption(doc.metadata["source"])
                st.markdown(doc.page_content[:200])
    # Storing assistant answer and sources in session state to display even after rerunning
    chat["messages"].append({"role": "assistant", "content": answer, "sources": [{"file": doc.metadata["source"], "preview": doc.page_content[:200]} for doc in sources]})