from timings import time_it, logger
from settings import Settings
import google.generativeai as genai
import time

class Responder:
    def __init__(self):
        # Configure Gemini API
        genai.configure(api_key=Settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.max_tokens = 8192  # Treat as a max character limit for prompt trimming

    def _trim_context(self, context, query, max_chars):
        # Build a header to determine available prompt space
        sys_prompt = "You are a knowledgeable staff member at the University of North Dakota."
        query_prompt = f"Question: {query}\nProvide a precise and complete answer."
        header = sys_prompt + "\n" + query_prompt + "\n"
        available_chars = max_chars - len(header) - 1000  # Reserve space for answer generation
        if len(context) > available_chars:
            context = context[:available_chars]
        return context

    @time_it
    def respond(self, query, context, memory):
        max_retries = 3
        retry_delay = 10

        for attempt in range(max_retries):
            try:
                trimmed_context = self._trim_context(context, query, self.max_tokens)
                memory_context = memory.get_context()

                prompt = (
                    "You are a knowledgeable staff member at the University of North Dakota.\n\n"
                    "Context:\n" + trimmed_context + "\n\n"
                    "Question:\n" + query + "\n\n"
                    "Conversation History:\n" + memory_context + "\n\n"
                    "Please provide a direct, concise, and accurate answer."
                )

                response = self.model.generate_content(prompt)
                if not response or not hasattr(response, 'text'):
                    raise Exception("No valid response from Gemini API.")

                answer = response.text.strip()
                logger.info(f"Generated response for: '{query}'")
                return answer
            except Exception as e:
                if "rate_limit_exceeded" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Error generating response: {str(e)}")
                    return "I'm unable to access that information right now."

        return "Technical issues encountered. Please try again later."
