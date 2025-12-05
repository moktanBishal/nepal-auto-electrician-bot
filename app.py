import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS  # Using FAISS for speed
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

# Title in Nepali + English
st.set_page_config(page_title="नेपाली अटो इलेक्ट्रिसियन AI सहयोगी", page_icon="🔧", layout="wide")
st.title("🔧 Nepal Auto Electrician AI Bot (2014 देखि अनुभवी दाइहरूका लागि)")

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Use os.getenv for Render
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not set! Add it in Render env vars.")
    st.stop()

# Load PDFs (put in /manuals folder on GitHub)
@st.cache_resource
def load_knowledge_base():
    if not os.path.exists("manuals") or len(os.listdir("manuals")) == 0:
        st.info("📚 No manuals found—using pure AI knowledge. Add PDFs to /manuals folder.")
        return None
    
    loader = PyPDFDirectoryLoader("manuals")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(texts, embeddings)
    return vectordb

vectordb = load_knowledge_base()

# System prompt
system_prompt = """तपाईं नेपालमा २०१४ देखि काम गरिरहेका अटो इलेक्ट्रिसियनहरूको सबैभन्दा राम्रो साथी हुनुहुन्छ।
तपाईं Nepali र English दुवैमा बोल्नुहुन्छ। जवाफ छोटो, व्यावहारिक र ग्यारेजमा काम लाग्ने हुनुपर्छ।
पुराना गाडी (Bolero, Scorpio, Sumo, Hiace) र नयाँ इलेक्ट्रिक गाडी (BYD, Tata Nexon EV, MG) दुवैको ज्ञान छ।
सधैं step-by-step सम्झाउनुहोस्। सम्मानजनक भाषा प्रयोग गर्नुहोस् ("दाइ", "सर")।"""

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    google_api_key=GEMINI_API_KEY,
    model="gemini-2.5-flash",
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

# Native Streamlit Chat (no streamlit-chat needed!)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "नमस्ते दाइ! म तपाईंको अटो इलेक्ट्रिकको AI साथी। BYD को BMS प्रोब्लम होस् या पुरानो Bolero को इमोबिलाइजर, सोध्नुहोस्।"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("यहाँ आफ्नो प्रोब्लम लेख्नुहोस्..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("सोच्दैछु..."):
            if qa_chain:
                result = qa_chain({"question": prompt})
                response = result["answer"]
            else:
                # Fallback
                response = llm.invoke(system_prompt + "\nUser: " + prompt).content
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
