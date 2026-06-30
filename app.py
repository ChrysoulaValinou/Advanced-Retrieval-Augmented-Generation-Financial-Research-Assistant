"""
app.py
======
Bonus Task A: Streamlit Chat Interface for Advanced RAG
"""

import streamlit as st
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.ingestion import build_vectorstore, build_documents
from src.retrieval import Retriever
from src.memory import MemoryManager

# Απενεργοποίηση ενοχλητικών logs
logging.getLogger("src.retrieval").setLevel(logging.WARNING)
logging.getLogger("src.memory").setLevel(logging.WARNING)
logging.getLogger("src.ingestion").setLevel(logging.WARNING)

# Ρύθμιση της σελίδας
st.set_page_config(page_title="Finance RAG Assistant", page_icon="📈", layout="wide")
st.title("🤖 Advanced RAG Chatbot")

# 1. Φόρτωση των συστημάτων (Cache για να μην φορτώνουν σε κάθε κλικ)
@st.cache_resource(show_spinner=False)
def init_rag():
    docs = build_documents()
    vs = build_vectorstore(force_rebuild=False)
    retriever = Retriever(vectorstore=vs, documents=docs)
    memory = MemoryManager(max_short_term_turns=6, llm_model="gemini-pro-latest")
    llm = ChatGoogleGenerativeAI(model="gemini-pro-latest", temperature=0.2)
    return retriever, memory, llm

with st.spinner("Φόρτωση βάσης δεδομένων και RAG pipeline..."):
    retriever, memory, llm = init_rag()

# 2. Αρχικοποίηση State για το Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_chunks" not in st.session_state:
    st.session_state.latest_chunks = []

# Base System Prompt
base_prompt = (
    "You are an expert AI research assistant. You answer questions based ONLY "
    "on the retrieved context provided to you. If the context does not contain "
    "the answer, say 'I don't know based on the provided documents.' "
    "Keep your answers clear, professional, and well-structured."
)

# 3. Εμφάνιση του Ιστορικού (Chat History)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Διαχείριση Νέου Μηνύματος
if user_input := st.chat_input("Ρώτησε κάτι για τα αρχεία σου..."):
    
    # Εμφάνιση ερώτησης χρήστη
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Παραγωγή Απάντησης
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # A. Ανάκτηση (Retrieval)
        try:
            docs_retrieved = retriever.hybrid(user_input, k=3)
            st.session_state.latest_chunks = docs_retrieved
            
            context_parts = []
            for d in docs_retrieved:
                source = getattr(d, 'source', 'Unknown')
                category = getattr(d, 'category', 'Unknown')
                text_content = getattr(d, 'text', getattr(d, 'content', getattr(d, 'preview', '')))
                context_parts.append(f"--- SOURCE: {source} ({category}) ---\n{text_content}")
            
            context_str = "\n\n".join(context_parts)
            if not context_str:
                context_str = "No relevant context found in the database."
        except Exception as e:
            context_str = "Error retrieving context."
            st.session_state.latest_chunks = []

        # B. Προετοιμασία LLM Messages
        system_prompt_text = memory.build_system_prompt(base_prompt)
        messages = [SystemMessage(content=system_prompt_text)]
        
        context_msg = f"Use the following retrieved documents to answer the user's latest question.\n\n{context_str}"
        messages.append(SystemMessage(content=context_msg))

        # Προσθήκη Short-Term History
        history = memory.get_short_term_history()
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
                
        messages.append(HumanMessage(content=user_input))

        # C. Κλήση LLM
        with st.spinner("Σκέφτεται..."):
            try:
                response = llm.invoke(messages)
                bot_reply = response.content
            except Exception as e:
                bot_reply = f"Σφάλμα API: {e}"

        # Εμφάνιση και αποθήκευση απάντησης
        message_placeholder.markdown(bot_reply)
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        
        # Ενημέρωση Short-Term Memory
        memory.add_user_turn(user_input)
        memory.add_assistant_turn(bot_reply)

# 5. Sidebar: Εμφάνιση των ανακτημένων Chunks (Απαίτηση Bonus Task)
with st.sidebar:
    st.header("📄 Retrieved Context")
    st.write("Εδώ εμφανίζονται τα έγγραφα που χρησιμοποιήθηκαν για την τελευταία απάντηση:")
    
    if st.session_state.latest_chunks:
        for i, chunk in enumerate(st.session_state.latest_chunks, 1):
            source = getattr(chunk, 'source', 'Unknown')
            category = getattr(chunk, 'category', 'Unknown')
            text = getattr(chunk, 'text', getattr(chunk, 'content', getattr(chunk, 'preview', '')))
            
            with st.expander(f"Chunk {i}: {source}"):
                st.caption(f"**Category:** {category}")
                st.caption(f"**Source:** {source}")
                st.write(text)
    else:
        st.info("Κάνε μια ερώτηση για να δεις τα σχετικά έγγραφα.")
        
    if st.button("🗑️ Εκκαθάριση Μνήμης & Λήξη Session"):
        memory.end_session()
        st.session_state.messages = []
        st.session_state.latest_chunks = []
        st.rerun()