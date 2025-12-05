import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# ====== Page Config ======
st.set_page_config(page_title="नेपाली अटो इलेक्ट्रिसियन AI", page_icon="Wrench", layout="centered")
st.markdown("""
    <h1 style='text-align: center; color: #d32f2f;'>Nepal Auto Electrician AI Bot</h1>
    <p style='text-align: center; font-size:18px;'>2014 देखि काम गर्ने दाइहरूका लागि पुरानो + नयाँ EV गाडीको साथी</p>
    <hr>
""", unsafe_allow_html=True)

# ====== API Key ======
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY हराइरहेको छ! Render मा Environment Variables मा थप्नुहोस्।")
    st.stop()

# ====== LLM ======
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.3
)

# ====== Chat History ======
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "नमस्ते दाइ! म तपाईंको अटो इलेक्ट्रिक AI साथी। Bolero को इमोबिलाइजर होस् या BYD Atto 3 को BMS, जे पनि सोध्नुहोस्!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ====== User Input ======
if prompt := st.chat_input("यहाँ आफ्नो समस्या लेख्नुहोस्..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("सोच्दैछु दाइ..."):
            system_prompt = "तपाईं नेपालका अनुभवी अटो इलेक्ट्रिसियन दाइहरूको सबैभन्दा राम्रो साथी हुनुहुन्छ। Nepali मा छोटो, व्यावहारिक र ग्यारेजमा काम लाग्ने जवाफ दिनुहोस्। सधैं step-by-step सम्झाउनुहोस् र 'दाइ' वा 'सर' भनेर सम्मान गर्नुहोस्।"
            response = llm.invoke(system_prompt + "\nप्रश्न: " + prompt).content
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
