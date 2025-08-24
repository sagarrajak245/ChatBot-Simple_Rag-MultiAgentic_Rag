from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")

# prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI customer support agent. Answer the questions as truthfully as possible. If you don't know the answer, say 'I don't know'"),
    ("human", "Question: {question}"),
])

# streamlit
st.title("Chatbot")

question = st.text_input("Enter your question:")

# model
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

output_parser = StrOutputParser()

chain = prompt | model | output_parser

if question:   
    try:
        answer = chain.invoke({"question": question})
        st.write(answer)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")