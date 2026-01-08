from flask import Flask, request, jsonify , render_template
from src.helper import load_pdf_files, text_split, download_embeddings, filter_to_minimal_docs
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
import os
from src.prompt import system_prompt
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate 
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embeddings = download_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


chatmodel = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="AIzaSyBbobAd8zgt6D_qK8BykVGj0y_2k_Oct90",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(chatmodel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])
    return str (response["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000 ,debug=True)

