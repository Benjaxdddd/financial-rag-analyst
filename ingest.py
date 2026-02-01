import os
import nest_asyncio
from dotenv import load_dotenv

# LlamaIndex Dependencias

from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

# Entorno

load_dotenv()
nest_asyncio.apply()

# Embedding Config 

print("Inicializando Embedding local")
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu"
)

# LlamaIndex Config Embbeding

Settings.embed_model = embed_model

def main():
    print("LlamaParse ---- (Markdown mode)")
    parser = LlamaParse(
        result_type="markdown",
        verbose=True,
        language="en",
    )

    file_extractor = {".pdf": parser}

    # consulta a data

    print("leyendo data.....")
    documents = SimpleDirectoryReader(
        "./data",
        file_extractor=file_extractor
    ).load_data()

    print(f"📄 Procesados {len(documents)} fragmentos/páginas.")

    #carga a chroma

    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("finance_portfolio")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    print("chromaDB ready my friendddddd")

if __name__ == "__main__":
    if not os.path.exists("./data"):
        os.makedirs("./data")
        print("no creaste la carpeta master")
    else:
        if len(os.listdir("./data")) == 0:
            print("no cargaste el pdf master")
        else:
            main()
