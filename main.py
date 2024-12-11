import streamlit as st
import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings

# Streamlit page configuration
st.set_page_config(page_title="RAG Chatbot from DataFrame")
st.header("RAG Chatbot from DataFrame")
OPENAI_API_KEY = st.text_input("OPENAI_APIKEY")

# Load and preprocess the DataFrame
def preprocess_dataframe(df):
    """Converts a DataFrame into a mapping of questions to answers."""
    mapping = {}
    for idx, row in df.iterrows():
        mapping[row["core_question"].strip()] = row["core_answer"].strip()
    return mapping

# Cache the retriever to avoid recomputation
@st.cache_resource
def create_retriever(question_answer_mapping):
    """Creates a retriever from the question-answer mapping using OpenAI embeddings."""
    documents = [
        Document(page_content=answer, metadata={"core_question": question})
        for question, answer in question_answer_mapping.items()
    ]
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    return retriever

# Chain creation
@st.cache_resource
def create_chain(_retriever):
    """Creates a Conversational Retrieval Chain with RAG-only responses using OpenAI API."""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Specify the LLM and its settings
    llm = ChatOpenAI(
        model_name="gpt-4o",  # Use "gpt-4" if you have access and need better performance
        temperature=0,  # Adjust the temperature as needed
        openai_api_key=OPENAI_API_KEY  # Replace with your OpenAI API key
    )
    
    # Create the chain with the custom prompt
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=_retriever, 
        memory=memory
    )
    
    return chain

# Load the predefined file
df = pd.read_excel("/Users/sirabhobs/Downloads/HR Helpdesk_Q&A Chatbot.xlsx", sheet_name="FAQ")

required_columns = ["main_node", "responsible_team", "question_tag", "core_question", "core_answer"]
if all(column in df.columns for column in required_columns):
    st.write("Preview of the DataFrame:")
    st.write(df)

    question_answer_mapping = preprocess_dataframe(df)
    retriever = create_retriever(question_answer_mapping)
    chain = create_chain(retriever)

    # Chat functionality
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "ฉันคือผู้ช่วยสุดยอดเยี่ยม ถามอะไรตอบได้ ไม่รู้ฉันก็จะตอบให้"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("ถามเลยค่า")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.messages.append({"role": "user", "content": user_input})

        response = chain.run(user_input)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.error("The uploaded file does not contain the required columns.")