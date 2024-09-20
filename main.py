import os
import langchain
from langchain import llms
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from io import BytesIO

import streamlit as st
OpenAI_Key=os.environ.get('OPENAI_API_KEY')

if __name__=="__main__":
    st.title("Chatboat For PDF")
    uploaded_file=st.file_uploader("Please upload the file here for discuss with it.",type="pdf")
    if uploaded_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
            loader=PyPDFLoader("temp.pdf")
            docs=loader.load()

            splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks=splitter.split_documents(docs)
    
            embedding_model= OpenAIEmbeddings(openai_api_key=OpenAI_Key)
   
            db=Chroma.from_documents(chunks,embedding_model, persist_directory="db")

            retrival=db.as_retriever()
    
            llm=ChatOpenAI(openai_api_key=OpenAI_Key)
    
            qa_chain=RetrievalQA.from_chain_type(llm=llm, retriever=retrival)
            query=st.text_input("Now you can ask me the question based on the uploaded file.")
            result=qa_chain.invoke(query)
            st.write(result)
