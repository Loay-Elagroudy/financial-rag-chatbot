import streamlit as st
import pandas as pd
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_groq import ChatGroq

# إعدادات واجهة المستخدم
st.set_page_config(page_title="Complaints Analytics RAG", page_icon="🤖", layout="centered")
st.title("🤖 Consumer Complaints RAG Chatbot")
st.markdown("### Milestone 5 — Live Analytics Dashboard (Ultra-Fast Mode)")

# سحب المفتاح السري المظبوط من الـ Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def init_resources():
    # 1. قراءة الداتا النظيفة والمصغرة
    df = pd.read_csv("mini_processed_corpus.csv").head(5000)
    df = df.dropna(subset=['rag_document'])
    
    documents = []
    for idx, row in df.iterrows():
        doc = Document(
            page_content=str(row['rag_document']),
            metadata={"complaint_id": str(row['Complaint ID']), "company": str(row['Company'])}
        )
        documents.append(doc)
    
    # 2. بناء Retriever سريع جداً ومستقر
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = 3
    
    # 3. استدعاء موديل Llama 3 عبر Groq بالمفتاح الصحيح
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )
    return retriever, llm

try:
    retriever, llm = init_resources()
    st.success("🚀 RAG System Active & Database Connected Instantly!")
except Exception as e:
    st.error(f"Initialization Error: {e}")

# إدارة شات البوت
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Type your question here..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        with st.spinner("Searching database & generating answer via Groq..."):
            docs = retriever.invoke(user_query)
            context = "\n\n".join(doc.page_content for doc in docs)
            
            full_prompt = (
                f"You are an expert customer support AI assistant specializing in analyzing consumer financial complaints.\n"
                f"Use the provided pieces of retrieved context to answer the user's question accurately.\n"
                f"If you cannot find the answer in the context, clearly state that you do not know.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {user_query}\n"
                f"Answer:"
            )
            
            try:
                response = llm.invoke(full_prompt)
                answer = response.content
            except Exception as e:
                answer = f"Error generating response: {e}"
                
            st.markdown(answer)
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
