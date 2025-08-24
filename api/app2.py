from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langserve import add_routes
import uvicorn
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()




os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
app=FastAPI(
    title="server",
    version="1.0"
)
# Update this line to include the model parameter
add_routes(
    app, ChatGoogleGenerativeAI(model="gemini-pro"),
    path="/gemini"
    
)

# Update this line as well
model=ChatGoogleGenerativeAI(model="gemini-1.5-pro")

model2=Ollama(model="llama2")

chatprompt1=ChatPromptTemplate.from_messages( "write me note {topic} in 10 word")

chatprompt2=ChatPromptTemplate.from_messages("write me heading {topic} in 15 word")

#route for model1 gemini
add_routes(app,chatprompt1,path="/note")

#route for model2 ollama
add_routes(app,chatprompt2,path="/heading")

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)