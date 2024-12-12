import streamlit as st
import pandas as pd
from module.agent import helpdeskAgent
from module.model.retriever import retriever

# Streamlit page configuration
st.set_page_config(page_title="RAG Chatbot from DataFrame", layout="centered")
st.header("RAG Chatbot from DataFrame")

# Load the predefined file and display it
try:
    file_path = "/Users/sirabhobs/Downloads/HR Helpdesk_Q&A Chatbot.xlsx"
    df = pd.read_excel(file_path, sheet_name="FAQ")
    df.reset_index(inplace=True)
    st.write("### Preview of the DataFrame:")
    st.write(df)
except FileNotFoundError:
    st.error(f"File not found at {file_path}. Please check the path and try again.")
    st.stop()

# Create retriever and chain
try:
    st.write("### Creating embeddings...")
    faiss_retriever = retriever(df)
    st.success("Embeddings created successfully!")
    agent = helpdeskAgent(faiss_retriever)
    
except Exception as e:
    st.error(f"Error creating retriever or chain: {e}")
    st.stop()

# Initialize chat state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "สวัสดีค่ะ! ฉันคือผู้ช่วยของคุณ ถามมาได้เลยค่ะ!"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("ถามอะไรมาได้เลยค่า...")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response from the agent
    try:
        print(user_input)
        response = agent.response(user_input)
    except Exception as e:
        response = f"ขออภัยค่ะ เกิดข้อผิดพลาด: {e}"

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})