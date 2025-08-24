from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI customer support agent. "
               "Answer the questions as truthfully as possible. "
               "If you don't know the answer, say 'I don't know'."),
    ("human", "Question: {question}")
])

# Streamlit UI
st.title("Groq Q&A Chatbot")
question = st.text_input("Enter your question:")

# Model
model = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)

# Output parser
output_parser = StrOutputParser()

# Chain
chain = prompt | model | output_parser

# Handle Q&A
if question:
    try:
        answer = chain.invoke({"question": question})
        st.write(answer)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
