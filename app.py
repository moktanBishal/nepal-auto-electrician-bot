import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI  # New: Gemini LLM
import os
from dotenv import load_dotenv

load_dotenv()

# Title in Nepali + English
st.set_page_config(page_title="नेपाली अटो इलेक्ट्रिसियन AI सहयोगी", page_icon="🔧")
st.title("🔧 Nepal Auto Electrician AI Bot (2014 देखि अनुभवी दाइहरूका लागि)")

# Gemini API Key (free from aistudio.google.com)
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]  # Change from GROQ

# Load PDFs (you will put Nepali/English manuals in /manuals folder)
@st.cache_resource
def load_knowledge_base():
    if not os.path.exists("manuals") or len(os.listdir("manuals")) == 0:
        return None
    
    loader = PyPDFDirectoryLoader("manuals")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(texts, embeddings, persist_directory="db")
    return vectordb

vectordb = load_knowledge_base()

# System prompt in Nepali + English (very important for tone)
system_prompt = """तपाईं नेपालमा २०१४ देखि काम गरिरहेका अटो इलेक्ट्रिसियनहरूको सबैभन्दा राम्रो साथी हुनुहुन्छ।
तपाईं Nepali र English दुवैमा बोल्नुहुन्छ। जवाफ छोटो, व्यावहारिक र ग्यारेजमा काम लाग्ने हुनुपर्छ।
पुराना गाडी (Bolero, Scorpio, Sumo, Hiace) र नयाँ इलेक्ट्रिक गाडी (BYD, Tata Nexon EV, MG) दुवैको ज्ञान छ।
सधैं step-by-step सम्झाउनुहोस्। सम्मानजनक भाषा प्रयोग गर्नुहोस् ("दाइ", "सर")।"""

# New: Gemini LLM (gemini-2.5-flash for speed/multimodal)
llm = ChatGoogleGenerativeAI(
    google_api_key=GEMINI_API_KEY,  # Change from Groq
    model="gemini-2.5-flash",  # Or "gemini-2.5-pro" for deeper reasoning (slower)
    temperature=0.3
)

memory = ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)

if vectordb:
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        combine_documents_chain_kwargs={"prompt": system_prompt}
    )
else:
    qa_chain = None

# Chat interface
from streamlit_chat import message

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "नमस्ते दाइ! म तपाईंको अटो इलेक्ट्रिकको AI साथी। BYD को BMS प्रोब्लम होस् या पुरानो Bolero को इमोबिलाइजर, सोध्नुहोस्।"})

for msg in st.session_state.messages:
    message(msg["content"], is_user=(msg["role"] == "user"), key=str(len(st.session_state.messages)) + msg["role"])

user_input = st.chat_input("यहाँ आफ्नो प्रोब्लम लेख्नुहोस्...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    message(user_input, is_user=True)

    with st.spinner("सोच्दैछु..."):
        if qa_chain:
            result = qa_chain({"question": user_input})
            response = result["answer"]
        else:
            # Fallback if no manuals uploaded
            response = llm.invoke(system_prompt + "\nUser: " + user_input).content

    st.session_state.messages.append({"role": "assistant", "content": response})
    message(response, is_user=False)
