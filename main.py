"""
main.py
=======
Entry point for the Conversational RAG Loop.
"""

import sys
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Local imports
from src.ingestion import build_vectorstore, build_documents
from src.retrieval import Retriever
from src.memory import MemoryManager

# Απενεργοποιούμε τα περιττά logs για να είναι καθαρό το chat
logging.getLogger("src.retrieval").setLevel(logging.WARNING)
logging.getLogger("src.memory").setLevel(logging.WARNING)
logging.getLogger("src.ingestion").setLevel(logging.WARNING)

def main():
    print("\n" + "="*60)
    print(" 🚀 Initialising Advanced RAG Chatbot...")
    print("="*60)

    # 1. Initialize Retriever (Loads documents and existing ChromaDB index)
    try:
        print("   [1/2] Loading documents for Hybrid Search...")
        docs = build_documents()
        
        print("   [2/2] Loading existing Vector Store...")
        vs = build_vectorstore(force_rebuild=False)
        
        retriever = Retriever(vectorstore=vs, documents=docs)
        print(" [✓] Retriever and Vector store loaded successfully.")
    except Exception as e:
        print(f"\n[!] Error loading vector store: {e}")
        print("Did you run 'python -m src.ingestion' first?")
        sys.exit(1)

    # 2. Initialize Memory Manager (Αλλάξαμε το μοντέλο σε gemini-pro)
    memory = MemoryManager(max_short_term_turns=6, llm_model="gemini-2.5-flash")
    print(" [✓] Memory layers (Short-term & Long-term) initialised.")

    # 3. Initialize the Google Gemini LLM (Αλλάξαμε το μοντέλο σε gemini-pro)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    print(" [✓] LLM (Gemini Pro) ready.")

    # Base System Prompt
    base_prompt = (
        "You are an expert AI research assistant. You answer questions based ONLY "
        "on the retrieved context provided to you. If the context does not contain "
        "the answer, say 'I don't know based on the provided documents.' "
        "Keep your answers clear, professional, and well-structured."
    )

    # Inject Long-Term Memory
    system_prompt_text = memory.build_system_prompt(base_prompt)

    print("\n" + "-"*60)
    print(" Type your questions below. Type 'exit' or 'quit' to end the session.")
    print(" (Ending the session will save your conversation to long-term memory)")
    print("-" * 60 + "\n")

    # ── Conversational Loop ──────────────────────────────────────────────────
    while True:
        try:
            user_input = input("🗣️ You: ").strip()
        except (KeyboardInterrupt, EOFError):
            user_input = "exit"

        if user_input.lower() in ['exit', 'quit']:
            print("\n[!] Ending session. Saving summary to Long-Term Memory...")
            memory.end_session()
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        print("   🔍 Searching knowledge base...")
        
        # A. Retrieve Context
        try:
            docs_retrieved = retriever.hybrid(user_input, k=3)
            context_parts = []
            
            for d in docs_retrieved:
                # Προσαρμογή στο δικό σου αντικείμενο RetrievedDoc!
                source = getattr(d, 'source', 'Unknown')
                category = getattr(d, 'category', 'Unknown')
                
                # Παίρνουμε το κείμενο είτε λέγεται text, είτε content, είτε preview
                text_content = getattr(d, 'text', getattr(d, 'content', getattr(d, 'preview', '')))
                
                context_parts.append(f"--- SOURCE: {source} ({category}) ---\n{text_content}")
            
            context_str = "\n\n".join(context_parts)
            if not context_str:
                context_str = "No relevant context found in the database."
        except Exception as e:
            print(f"   [!] Retrieval Error: {e}")
            context_str = "Error retrieving context. Answer based on memory if possible."

        # B. Construct the Messages
        messages = []
        messages.append(SystemMessage(content=system_prompt_text))
        
        context_msg = (
            f"Use the following retrieved documents to answer the user's latest question.\n\n"
            f"{context_str}"
        )
        messages.append(SystemMessage(content=context_msg))

        history = memory.get_short_term_history()
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
                
        messages.append(HumanMessage(content=user_input))

        # C. Call LLM
        print("   ✍️ Generating answer...")
        try:
            response = llm.invoke(messages)
            bot_reply = response.content
        except Exception as e:
            bot_reply = f"Sorry, I encountered an error with the LLM API: {e}"

        print(f"\n🤖 Assistant:\n{bot_reply}\n")

        # D. Update Short-Term Memory
        memory.add_user_turn(user_input)
        memory.add_assistant_turn(bot_reply)

if __name__ == "__main__":
    main()