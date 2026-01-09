from flask import Flask, request, jsonify, render_template
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from src.prompt import system_prompt
import os

app = Flask(__name__)

# 1. Load Keys
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# 2. Setup Embeddings & Index
embeddings = download_embeddings()
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# 3. Setup Model
chatmodel = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="AIzaSyD5jayqxAH_f-_ar1WIxmh6TdyHARa0dT4",
    temperature=0
)

# --- NEW MEMORY LOGIC STARTS HERE ---

# A. Create a "History Aware" Retriever
# This sub-chain rephrases the latest question given the chat history
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    chatmodel, retriever, contextualize_q_prompt
)

# B. Create the Answer Chain (QA)
# This chain actually answers the question using the docs
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("placeholder", "{chat_history}"), # <--- History goes here
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatmodel, qa_prompt)

# C. Combine them into the RAG Chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# D. Managing Chat History State
# We store history in a simple dictionary for this demo
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# E. The Final Chain with Automatic History Management
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# --- ROUTES ---

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(f"User Input: {input}")
    
    # We use a static session ID for simplicity in this demo.
    # In a real app with login, this would be the user's ID.
    session_id = "default_user_session"
    
    response = conversational_rag_chain.invoke(
        {"input": msg},
        config={"configurable": {"session_id": session_id}}
    )
    
    print("Response:", response["answer"])
    return str(response["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)