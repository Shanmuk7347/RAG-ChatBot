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
    expanded_query = chain.invoke(query)
    content = expanded_query
    return content

if __name__ == "__main__":
    queries =  ["what is vectorstore and comapre it with SQL", "what is attnetion", "What is the info in page 7"]
    for query in queries:
        response = expand_query(query)
        print(f"query: {query}, Response: {response}")