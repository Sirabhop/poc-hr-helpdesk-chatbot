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
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Login state management
try:
    authenticator.login()
except stauth.utilities.LoginError as e:
    st.error(e)
    
if st.session_state["authentication_status"]:
    authenticator.logout('Logout', 'sidebar')

    st.header("RAG Chatbot from DataFrame")

    # Load the predefined file and display it
    @st.cache_data
    def load_data(file_path, sheet_name):
        return pd.read_excel(file_path, sheet_name=sheet_name)

    try:
        file_path = "/Users/sirabhobs/Downloads/HR Helpdesk_Q&A Chatbot.xlsx"
        df = load_data(file_path, "FAQ")
        df.reset_index(inplace=True)
        st.write("### Preview of the DataFrame:")
        st.dataframe(df)
    except FileNotFoundError:
        st.error(f"File not found at {file_path}. Please check the path and try again.")
        st.stop()

    # Create retriever and chain
    @st.cache_resource
    def initialize_agent(dataframe):
        faiss_retriever = retriever(dataframe)
        return helpdeskAgent(faiss_retriever)

    try:
        st.write("### Creating embeddings...")
        agent = initialize_agent(df)
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

    # Display chat and handle input
    display_chat()
    handle_user_input()

elif st.session_state["authentication_status"] == False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')