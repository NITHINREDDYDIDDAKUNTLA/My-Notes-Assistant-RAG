import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_classic.vectorstores import FAISS
import os
from dotenv import load_dotenv
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
load_dotenv()

#PROJECT TITLE
st.title("My Notes Assistant - RAG Chatbot")
st.caption("Ask anything from your PDF notes - Powered by RAG")

#DOCUMENT LOADER
document = PyPDFLoader("sample_notes.pdf") # replace sample_notes.pdf with your notes.pdf
doc=document.load()
# print(len(doc))

#TEXT SPLITTER

splitter=RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=100)
chunks=splitter.split_documents(doc)
# print(len(chunks))

#EMBEDDING

embeddings=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

#VECTOR DATABASE

faiss=FAISS.from_documents(chunks,embeddings)
vectors=faiss.as_retriever(search_kwargs={"k":3})

#MODEL CREATION (GROQ)
model=ChatGroq(model="qwen/qwen3.8-27b",temperature=0,groq_api_key=os.getenv("GROQ_API_KEY"))

#RETRIEVAL

chain=RetrievalQA.from_chain_type(llm=model,retriever=vectors,chain_type="stuff",return_source_documents=True)

#USER INPUT QUERY
query=st.chat_input("Ask from your notes...")

if query:
    result=chain.invoke({"query":query})
    st.write("***ANSWER***")
    st.write(result["result"])
    st.write("***SOURCE***")
    st.write(result["source_documents"][0].page_content[:300])
