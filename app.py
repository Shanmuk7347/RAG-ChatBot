import streamlit as st
from rag_chain import get_chain, get_retriever, delete_chat_collection, get_vectorstore
from ingest import build_vector_data
import os
import uuid
import shutil
from database import create_tables, save_chat, load_chats, delete_chat, save_message, load_messages, save_documents, load_documents
import json


# Setup app for RAG-Chatbot
st.set_page_config(
    page_title="RAG-Chatbot",
    page_icon="🤖",
    layout="centered"
)

create_tables()

def new_chat(name="New Chat"):
    return {
        "name": name,
        "id": uuid.uuid4().hex[:12],
        "docs": [],
        "messages": []
    }

st.title("🔍 Ask Your PDFs")
st.caption("Powered by Llama, ChromaDB, and Streamlit")
st.markdown("---")

# setting up session state and loading exixting data 
if "chats" not in st.session_state:
    saved_chats = load_chats()
    if saved_chats:
        for c in saved_chats:
            c["messages"] = load_messages(c["id"])
            c["docs"] = load_documents(c["id"])
        st.session_state.chats = saved_chats
    else:    
        first = new_chat("First Chat")
        st.session_state.chats = [first]
        save_chat(chat_id=first["id"],name=first["name"])

if "chat_count" not in st.session_state:
    st.session_state.chat_count = len(st.session_state.chats)

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
        st.session_state.chat_count += 1
        new = new_chat(f"Chat {st.session_state.chat_count}")
        st.session_state.chats.append(new)
        save_chat(chat_id=new["id"],name=new["name"])
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
                save_documents(chat["id"], file.name)
                new_file_paths.append(file_path) 

            if new_file_paths:
                with st.spinner("Ingesting files..."):
                    try:
                        build_vector_data(file_paths=new_file_paths, chat_id=chat["id"])
                        get_vectorstore.clear()
                    except Exception as e:
                        st.error(f"Error during ingestion: {e}")
                st.success("Ingestion complete!")
                st.rerun()
            else:
                st.info("No new files for ingestion.")

    st.markdown("---")

    if st.button("🚮Delete This chat"):
        with st.spinner("Deleting chat and associated data..."):
            if os.path.exists(f"./Docs/{chat["id"]}"):
                shutil.rmtree(f"./Docs/{chat["id"]}")
            delete_chat_collection(chat["id"])
            delete_chat(chat["id"])
            get_vectorstore.clear()
            st.session_state.chats.remove(chat)
        if not st.session_state.chats:
            new = new_chat("New Chat")
            st.session_state.chats.append(new)
            save_chat(chat_id=new["id"],name=new["name"])
        st.session_state.active_chat_id = st.session_state.chats[0]["id"]
        st.rerun()
        
for message in chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    
        if (message["role"] == "assistant" and message.get("sources")):
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
    save_message(chat["id"], "user", prompt, None)

# Invoke retriever and chain to get the source and answers
    with st.spinner("Searching through the docs..."):
        sources = get_retriever(chat["id"]).invoke(prompt)
        

# Displaying the answer and sources
    response = ""
    chain = get_chain(chat["id"])
    with st.chat_message("assistant"):
        placeholder = st.empty()
        for chunk in chain.stream(prompt):
            response += chunk
            placeholder.markdown(response + "|")
        placeholder.markdown(response)
        with st.expander("Sources"):
            for doc in sources:
                st.caption(doc.metadata["source"])
                st.markdown(doc.page_content[:200])
    # Storing assistant answer and sources in session state to display even after rerunning
    SOURCES = [{"file": doc.metadata["source"], "preview": doc.page_content[:200]} for doc in sources]
    chat["messages"].append({"role": "assistant", "content": response, "sources": SOURCES})
    save_message(chat["id"], "assistant", response, json.dumps(SOURCES))