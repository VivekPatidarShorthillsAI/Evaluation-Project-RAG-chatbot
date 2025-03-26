# src/ui.py
import streamlit as st
from RAG_pipeline.responder import Responder
from src.query_handler import QueryHandler
import logging

class ChatUI:
    def __init__(self):
        self.responder = Responder()
        self.query_handler = QueryHandler()
        logging.basicConfig(filename='assets/chatbot.log', level=logging.INFO)
        
    def run(self):
        st.title("RAG Chatbot")
        st.write("Ask questions based on the crawled content")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        # Accept user input
        if prompt := st.chat_input("What would you like to know?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                try:
                    # Get relevant context
                    results = self.query_handler.process_query(prompt)
                    context = "\n\n".join([res[2] for res in results])
                    
                    # Generate response
                    response = self.responder.generate_response(prompt, context)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = "Sorry, I encountered an error processing your request."
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    logging.error(f"UI Error: {str(e)}")