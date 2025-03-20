from app import app
from waitress import serve

def initialize_chatbot():
    # (Optional initialization tasks can be placed here)
    print("Chatbot initialized.")

if __name__ == '__main__':
    initialize_chatbot()
    app.run(debug=True)
