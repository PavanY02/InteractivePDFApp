import os
from langchain import PromptTemplate
from langchain_community.vectorstores import FAISS

import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
Generate answers from the data in the vector store itself only if it's there in the document; otherwise, say "There is no information about it"."""

instruction = "Answer the Question You asked: \n\n {text}"

SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS

template = B_INST + SYSTEM_PROMPT + instruction + E_INST
prompt = PromptTemplate(template=template, input_variables=["text"])

def make_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(raw_text)
    return chunks

def make_vector_store(textChunks):
    embeddings = HuggingFaceHubEmbeddings(huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    vector_store = FAISS.from_texts(textChunks, embedding=embeddings)
    return vector_store

def make_conversational_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        verbose=True,
        temperature=0.2,
        google_api_key=os.getenv("Google_API_KEY"),
        max_output_tokens=500  # Adjust this value as needed
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        prompt=prompt,
        return_messages=True
    )
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversational_chain

def user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chatHistory = response["chat_history"]
    for i, message in enumerate(st.session_state.chatHistory):
        # Clean up the message content
        content = message.content.replace('\n', ' ').strip()
        st.write(f"{i}. {content}")

def main():
    st.set_page_config("Chat with multiple PDFs")
    st.header("Chat with Multiple PDFs 🐆")
    user_question = st.text_input("Enter your questions regarding the PDFs")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chatHistory = None
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("Settings")
        st.subheader("Upload Your Documents")
        pdf_docs = st.file_uploader("Upload your PDF files and click on Process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = make_pdf_text(pdf_docs)
                textChunks = text_chunks(raw_text)
                vector_store = make_vector_store(textChunks)
                st.session_state.conversation = make_conversational_chain(vector_store)
                st.success("Done")
        st.title("Credits to thala7️⃣")

if __name__ == "__main__":
    main()
