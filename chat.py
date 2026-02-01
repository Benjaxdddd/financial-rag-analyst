import os
import sys
import chromadb

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

load_dotenv()


if not os.path.exists("./chroma_db"):
    print("asegurate de haben ejecutado el ingest porque no se encuentra la carpeta chroma_db")
    sys.exit(1)

#modelos

print("local embedding cargando....")
embed_model=HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu"
)
Settings.embed_model=embed_model

# LLM

llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0
)
Settings.llm = llm

def start_chat_session():
    print("Accediendo a la bdd vectorial...")
    db = chromadb.PersistentClient(path="./chroma_db")

    # get a la coleccion 

    chroma_collection = db.get_or_create_collection("finance_portfolio")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # index

    Index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )

    # Chat engine 

    chat_engine = Index.as_chat_engine(
        chat_mode="context",
        system_prompt =(
            "Eres un Analista Financiero Senior llamado 'CoinBot'. "
            "Tu objetivo es responder preguntas sobre el reporte financiero proporcionado. "
            "Reglas Estrictas: "
            "1. Responde SOLO basándote en el contexto recuperado a continuación. "
            "2. Si la respuesta no está en el contexto, di: 'No encuentro esa información en el documento'. "
            "3. en caso de que el usuario acceda un input en español debes buscar el simil en ingles'. "
            "4. Cita la sección o página si es posible. "
            "5. Mantén un tono profesional y directo."
        ),
        similarity_top_k=3,
    )
    print("\n el sistema se encuentra inicializado. Pregunta sobre el reporte financiero (para salir escribe salir)")
    print("-"*50)

    while True:
        user_input = input("\nUsuario 🗣️: ")
        
        if user_input.lower() in ['salir', 'exit', 'quit']:
            print("👋 Cerrando sesión.")
            break
            
        if not user_input.strip():
            continue

        # Generación de Respuesta
        try:
            # stream_chat 
            streaming_response = chat_engine.stream_chat(user_input)
            
            print("CoinBot: ", end="", flush=True)
            for token in streaming_response.response_gen:
                print(token, end="", flush=True)
            print("\n")
            
            print("🔍 [Evidencia Recuperada - Transparencia RAG]")
            for i, node in enumerate(streaming_response.source_nodes):
                # Score de similitud (entre 0 y 1, donde 1 es idéntico)
                score = node.score if node.score else 0.0
                # Primeros 100 caracteres del fragmento usado
                print(f"   Fuente #{i+1} (Similitud: {score:.3f}): {node.node.get_text()[:80].replace(chr(10), ' ')}...")

        except Exception as e:
            print(f"❌ Error al generar respuesta: {e}")

if __name__ == "__main__":
    start_chat_session()

