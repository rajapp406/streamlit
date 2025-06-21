import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage
load_dotenv()

st.title("Chatbot with Memory and Trimmed Messages")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [] 
if "trimmed_messages" not in st.session_state:
    st.session_state.trimmed_messages = []  

def trim_messages(messages, max_length=5):
    return messages[-max_length:] if len(messages) > max_length else messages

def azure_openai_model():
    return AzureChatOpenAI(
        model='gpt-4o',
        api_key=os.environ.get('OPENAI_API_KEY'),
        azure_endpoint="https://ai-proxy.lab.epam.com",
        api_version="2023-12-01-preview",
    )

def chat_with_azure(prompt):
    client = azure_openai_model()
    try:
        response = client.invoke(prompt)
        return response.content
    
    except Exception as e:
        return f"Error: {str(e)}"

def get_response(user_input):

    st.session_state.messages.append(HumanMessage(content=user_input))
    
    st.session_state.trimmed_messages = trim_messages(st.session_state.messages)
    
    response = chat_with_azure(st.session_state.trimmed_messages)
    st.session_state.messages.append(AIMessage(content=response))
    return response

st.write("Ask me anything! I can refer to past questions/answers when needed.")

user_input = st.text_input("Your question:", key="user_input")

if user_input:
    response = get_response(user_input)
    
    st.write("Chatbot:", response)

# Display conversation history
st.write("Conversation History:")
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        st.write(f"You: {message.content}")
    elif isinstance(message, AIMessage):
        st.write(f"Chatbot: {message.content}")