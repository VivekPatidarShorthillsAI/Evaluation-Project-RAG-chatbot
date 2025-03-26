# RAG_pipeline/responder.py
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging

load_dotenv()

class Responder:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        logging.basicConfig(filename='assets/chatbot.log', level=logging.INFO)
        
    def generate_response(self, query: str, context: str) -> str:
        """Generate response using Gemini with provided context"""
        try:
            logging.info(f"Generating response for query: {query}")
            prompt = f"""
            You are a knowledgeable staff member at the University of North Dakota.\n\n
            Use the following Context to answer the query at the end.
            Please provide a direct, concise, complete and accurate answer including facts and phrases.
            Give the answer in a declarative manner.
            Context: {context}
            
            Question: {query}
            
            Answer:
            """
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "I encountered an error while processing your request. Please try again."