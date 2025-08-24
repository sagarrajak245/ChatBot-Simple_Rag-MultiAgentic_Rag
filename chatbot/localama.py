from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core import prompts

from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")


#step1: prompt template

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI customer support agent. Answer the questions as truthfully as possible. If you don't know the answer, say 'I don't know'"),
    ("human", "Question: {question}"),
])





#step2: streamlit
st.title("Chatbot")

question = st.text_input("Enter your question:")

#step3: model
model = Ollama(model="llama2")

#step4: output parser
output_parser = StrOutputParser()

#step5: chain
chain = prompt | model | output_parser

if question:
    try:
        answer = chain.invoke({"question": question})
        st.write(answer)
    except Exception as e:
        st.write("Error:", e)
        



        























