import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from llama_index.core import GPTVectorStoreIndex, SimpleDirectoryReader
import openai
import string
import random

load_dotenv()

os.environ["OPENAI_PI_KEY"] = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)


def get_answer_openai(text):

    documents = SimpleDirectoryReader("data").load_data()
    index = GPTVectorStoreIndex.from_documents(documents)

    # Par défaut, les données sont stockées en mémoire. Pour les enregistrer sur disque (sous ./storage) :
    index.storage_context.persist()

    # Pour recharger à partir du disque :
    from llama_index.core import StorageContext
    from llama_index.core import load_index_from_storage

    # reconstruire le contexte de stockage
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # charger l'index
    index = load_index_from_storage(storage_context)

    # Pour effectuer une requête :
    query_engine = index.as_query_engine()
    response = query_engine.query(text)
    return response


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    text = request.form.get("text")
    if text:
        response = process_text(text)
        return jsonify({"text": response["text"]})
    return jsonify({"text": "Invalid request"})


def process_text(text):
    # Placeholder function for processing user's text input
    # Replace this with your own implementation
    return_text = get_answer_openai(text)
    # generating random strings
    res = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
    return {"text": str(return_text)}  # Convert return_text to a string


if __name__ == "__main__":
    app.run(debug=True)
