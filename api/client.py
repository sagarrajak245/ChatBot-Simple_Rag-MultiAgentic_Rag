import streamlit as st
import requests

def get_api_response(url, topic):
    try:
        response = requests.post(url, json={'input': {"topic": topic}})
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        response_json = response.json()
        
        if "output" not in response_json:
            return f"Error: Unexpected response structure. 'output' key not found. Response: {response_json}"
        
        output = response_json["output"]
        if "messages" in output:
            # Extract the last message content
            messages = output["messages"]
            if messages:
                return messages[-1]["content"]
            else:
                return "Error: No messages found in the output."
        elif "content" in output:
            return output["content"]
        else:
            return f"Error: Neither 'content' nor 'messages' key found in output. Output: {output}"
    except requests.RequestException as e:
        return f"Error: Failed to get response from API. {str(e)}"

def get_gemini_response(topic):
    return get_api_response("http://localhost:8000/note/invoke", topic)

def get_ollama_response(topic):
    return get_api_response("http://localhost:8000/heading/invoke", topic)

st.title("Gemini and Ollama Client")

topic = st.text_input("Enter a topic")

if st.button("Generate Note"):
    gemini_note = get_gemini_response(topic)
    st.write(gemini_note)

if st.button("Generate Heading"):
    ollama_heading = get_ollama_response(topic)
    st.write(ollama_heading)

# ... rest of the code remains unchanged ...


















