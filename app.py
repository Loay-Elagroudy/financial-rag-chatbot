import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from google import genai

# إعدادات واجهة المستخدم
st.set_page_config(page_title="Complaints Analytics RAG", page_icon="🤖", layout="centered")
st.title("🤖 Consumer Complaints RAG Chatbot")
st.markdown("### Milestone 5 — Live Analytics Dashboard")
st.write("Ask questions about companies, financial products, or specific consumer disputes.")

# الـ Key الشغال بتاعك
GOOGLE_API_KEY = "AQ.Ab8RN6L_XFsIs2rEVzid13Zm-7vFUKU62v5F6r2GUSO5u0La_g"

@st.cache_resource
def init_resources():
    # تحميل الـ DB والـ Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="./chromadb_index", embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # تحميل عميل جوجل الجديد
    client = genai.Client(api_key=GOOGLE_API_KEY)
    return retriever, client

try:
    retriever, client = init_resources()
    st.success("RAG System Active & Database Connected!")
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
        with st.spinner("Searching database & generating answer..."):
            # 1. الـ Retrieval
            docs = retriever.invoke(user_query)
            context = "\n\n".join(doc.page_content for doc in docs)
            
            # 2. الـ Prompt
            full_prompt = (
                f"You are an expert customer support AI assistant specializing in analyzing consumer financial complaints.\n"
                f"Use the provided pieces of retrieved context to answer the user's question accurately.\n"
                f"If you cannot find the answer in the context, clearly state that you do not know.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {user_query}\n"
                f"Answer:"
            )
            
            # 3. الـ Generation
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=full_prompt,
                )
                answer = response.text
            except Exception as e:
                answer = f"Error generating response: {e}"
                
            st.markdown(answer)
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
