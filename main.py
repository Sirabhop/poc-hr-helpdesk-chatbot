import streamlit as st
import pandas as pd
import streamlit_authenticator as stauth
import yaml

from module.agent import helpdeskAgent
from module.model.retriever import retriever

# Load configuration file
with open('./config.yaml') as file:
    config = yaml.load(file, Loader=yaml.loader.SafeLoader)

# Streamlit page configuration
st.set_page_config(page_title="RAG Chatbot from DataFrame", layout="centered")

# Initialize the authenticator
try:
    authenticator = stauth.Authenticate(
        credentials=config.get('credentials', {}),
        cookie_name=config.get('cookie', {}).get('name', 'default_cookie_name'),
        key=config.get('cookie', {}).get('key', 'default_cookie_key'),
        expiry_days=config.get('cookie', {}).get('expiry_days', 30),
    )
    
except KeyError as e:
    st.error(f"Missing key in configuration file: {e}")
    st.stop()

# Login state management
try:
    res_authentication = authenticator.login('main')
except stauth.utilities.LoginError as e:
    st.error(e)
    
if st.session_state["authentication_status"]:
    
    authenticator.logout('Logout', 'sidebar')

    st.header("RAG Chatbot from DataFrame")
    
    st.write(f"Welcome {st.session_state["name"]} 👋🏻")

    # Load the predefined file and display it
    @st.cache_data
    def load_data(file_path, sheet_name):
        return pd.read_excel(file_path, sheet_name=sheet_name)

    try:
        df = load_data("./data/faq.xlsx", "FAQ")
        df.reset_index(inplace=True)
    except FileNotFoundError:
        st.error("File not found. Please check the path and try again.")
        st.stop()

    # Create retriever and chain
    @st.cache_resource
    def initialize_agent(dataframe, preference):
        st.write('step1')
        faiss_retriever = retriever(dataframe)
        st.write('step2')
        return helpdeskAgent(faiss_retriever, preference)

    try:
        st.write("### Creating embeddings...")
        
        agent = initialize_agent(df, st.session_state["roles"])
        st.success("Embeddings created successfully!")
    except Exception as e:
        st.error(f"Error creating retriever or chain: {e}")
        st.stop()

    # Initialize chat state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "สวัสดีค่ะ! ฉันคือผู้ช่วยของคุณ ถามมาได้เลยค่ะ!"}
        ]

    # Chat interface
    def display_chat():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def handle_user_input():
        user_input = st.chat_input("ถามอะไรมาได้เลยค่า...")
        if user_input:
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Generate and display assistant response
            try:
                response = agent.run(user_input)
            except Exception as e:
                response = f"ขออภัยค่ะ เกิดข้อผิดพลาด: {e}"

            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    display_chat()
    handle_user_input()

elif st.session_state["authentication_status"] == False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')