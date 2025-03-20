import json
import time
from query_handler import QueryHandler
from responder import Responder

# Dummy database manager for demonstration.
# Replace with your actual DB manager if available.
class DummyDBManager:
    def search(self, query):
        # This should return a list of documents (each with a page_content attribute)
        # For now, we return an empty list.
        return []

# Dummy memory class for demonstration.
# Replace with your actual memory class if available.
class DummyMemory:
    def get_context(self):
        # Return any conversation history or context if available.
        return ""

def main():
    # Load the golden set of queries and ground-truth answers
    with open('Golden_set.json', 'r', encoding='utf-8') as infile:
        golden_set = json.load(infile)

    # Initialize your QueryHandler and Responder using your modules.
    db_manager = DummyDBManager()  # Use your actual DB manager if available.
    query_handler = QueryHandler(db_manager)
    responder = Responder()
    memory = DummyMemory()  # Use your actual memory instance if available.

    model_output = []

    # Process each query from Golden_set.json
    for item in golden_set:
        query = item["query"]
        print(f"Processing query: {query}")

        # Get context from the query handler (this uses your FAISS and chunked data)
        context = query_handler.handle(query, memory)

        # Get the response from your Responder (which calls the Gemini API)
        answer = responder.respond(query, context, memory)

        # Wait 15 seconds after each API call to avoid API key fetching errors
        time.sleep(15)

        # Append the result to the model_output list in the desired format
        model_output.append({
            "query": query,
            "ground_truth": answer
        })

    # Write the model output to Model_Output_Set.json
    with open('Model_Output_Set.json', 'w', encoding='utf-8') as outfile:
        json.dump(model_output, outfile, indent=4, ensure_ascii=False)

    print("Model_Output_Set.json has been created successfully.")

if __name__ == '__main__':
    main()
