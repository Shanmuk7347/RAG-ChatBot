from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from logger import logger
from config import settings
import streamlit as st
import json
from query.prompt import PROMPT
from retry import safe_invoke

Gemini_API_KEY = settings.gemini_api_key
OpenAI_API_KEY = settings.openai_api_key

@st.cache_resource(show_spinner=False)
def get_understanding_llm(provider=settings.default_provider, model= settings.default_model):
    logger.info(f"Initializing LLM, provider: {provider}, model: {model}")
    if provider == "Ollama":
        return ChatOllama(model=model, temperature=0)
    elif provider == "Google":
        return ChatGoogleGenerativeAI(model=model, api_key=Gemini_API_KEY, streaming=True)
    elif provider == "OpenAI":
        return ChatOpenAI(model=model, temperature=0.3, api_key=OpenAI_API_KEY, streaming=True)

prompt = PROMPT

def get_understanding_chain(provider: str=settings.default_provider, model: str= settings.default_model):
    llm = get_understanding_llm(provider, model)
    chain = ({"query": RunnablePassthrough()} | prompt | llm | JsonOutputParser())
    return chain

def expand_query(query: str, provider: str=settings.default_provider, model: str= settings.default_model):
    chain = get_understanding_chain(provider, model)
    expanded_query = safe_invoke(chain, query)

    # Fill in missing keys with defaults
    expanded_query.setdefault("query_type", "simple")
    expanded_query.setdefault("rewritten_query", query)
    expanded_query.setdefault("metadata_filter", {})
    expanded_query.setdefault("sub_queries", [])
    expanded_query.setdefault("confidence", 0.0)

    logger.info(f"query type: {expanded_query['query_type']}")
    logger.info(f"rewritten query: {expanded_query['rewritten_query']}")
    logger.info(f"metadata: {expanded_query['metadata_filter']}")
    logger.info(f"sub queries: {expanded_query['sub_queries']}")

    return expanded_query


if __name__ == "__main__":
    queries =  ["what is vectorstore and comapre it with SQL", "what is attnetion", "What is the info in page 7"]
    for query in queries:
        response = expand_query(query)
        print(f"query: {query}, Response: {response}")