import streamlit as st
import os
import pandas as pd
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from google import genai

# إعدادات واجهة المستخدم
st.set_page_config(page_title="Complaints Analytics RAG", page_icon="🤖", layout="centered")
st.title("🤖 Consumer Complaints RAG Chatbot")
st.markdown("### Milestone 5 — Live Analytics Dashboard")

# سحب الـ API Key من إعدادات سيرفر Streamlit الآمنة أو استخدام الـ Key المباشر كـ Fallback
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "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")

@st.cache_resource
def init_resources():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # قراءة الداتا
    df = pd.read_csv("mini_processed_corpus.csv").head(5000)
    df = df.dropna(subset=['rag_document'])
    
    documents = []
    for idx, row in df.iterrows():
        doc = Document(
            page_content=str(row['rag_document']),
            metadata={"complaint_id": str(row['Complaint ID']), "company": str(row['Company'])}
        )
        documents.append(doc)
    
    vector_store = Chroma.from_documents(documents=documents, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # هنا التعديل الجذري: تمرير المفتاح مباشرة جوه الكلاينت بدون الاعتماد على الـ Environment variables
    client = genai.Client(api_key=GOOGLE_API_KEY)
    return retriever, client

try:
    with st.spinner("Initializing Live Vector Database..."):
        retriever, client = init_resources()
    st.success("RAG System Active & Database Connected Live!")
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
                # استخدام الموديل المستقر القياسي
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=full_prompt,
                )
                answer = response.text
            except Exception as e:
                answer = f"Error generating response: {e}"
                
            st.markdown(answer)
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
