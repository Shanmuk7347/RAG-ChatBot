import streamlit as st
from rag_chain import retriever, chain, data

# Setup app for RAG-Chatbot
st.set_page_config(
    page_title="RAG-Chatbot",
    page_icon="🤖",
    layout="centered"
)
st.title("🔍 Ask Your PDFs")
st.caption("Powered by Llama")
st.markdown("---")

# setting up session state to store previous conversations so that they wont disapper due to rerunning
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversations
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.caption(source["file"])
                    st.markdown(source["preview"])

# User input    
if prompt := st.chat_input("Ask anything"):
    with st.chat_message(name="user"):
        st.markdown(prompt)
    # Storing user query in session state to display even after rerunning
    st.session_state.messages.append({"role": "user", "content": prompt})

# Invoke retriever and chain to get the answer and sources
    with st.spinner("Searching through the docs..."):
        sources = retriever.invoke(prompt)
        answer = chain.invoke(prompt)
    """ Retriver gives relavent documents along with the top 4 chunks that LLM uses as Context, whereas Chain gives the answer given by LLM """

# Displaying the answer and sources
    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("Sources"):
            for doc in sources:
                st.caption(doc.metadata["source"])
                st.markdown(doc.page_content[:200])
    # Storing assistant answer and sources in session state to display even after rerunning
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": [{"file": doc.metadata["source"], "preview": doc.page_content[:200]} for doc in sources]})