# 🏦 Financial Analyst AI - Enterprise RAG

> Un asistente de IA financiero "CoinBot" capaz de analizar reportes 10-K, citar fuentes exactas y evitar alucinaciones mediante filtrado vectorial.

## 📋 Descripción
Este proyecto implementa una arquitectura RAG (Retrieval-Augmented Generation):
1.  **Parsing Estructural:** Convierte PDFs complejos a Markdown para preservar tablas financieras.
2.  **Verificación de Hechos:** Inferencia basada estrictamente en contexto recuperado.
3.  **Seguridad Matemática:** Post-procesamiento de similitud para descartar información irrelevante antes de llegar al LLM.

## 🛠️ Stack Tecnológico
* **Orquestación:** LlamaIndex
* **LLM:** llama-3.3-70b-versatile (vía Groq LPU para baja latencia)
* **Embeddings:** BAAI/bge-small-en-v1.5 (Ejecución local/CPU)
* **Vector DB:** ChromaDB (Persistencia local)
* **Ingesta:** LlamaParse (Visión computacional para documentos)

## 🚀 Instalación y Uso

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/Benjaxdddd/financial-rag-analyst.git](https://github.com/Benjaxdddd/financial-rag-analyst.git)
    cd financial-rag-analyst
    ```

2.  **Configurar entorno:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Configurar Variables de Entorno:**
    Crea un archivo `.env` y añade tus claves:
    ```
    GROQ_API_KEY=gsk_...
    LLAMA_CLOUD_API_KEY=llx_...
    ```

4.  **Ejecutar Ingesta (ETL):**
    Coloca tu PDF en la carpeta `data/` y ejecuta:
    ```bash
    python ingest.py
    ```

5.  **Iniciar Chat:**
    ```bash
    python chat.py
    ```

## ⚖️ Licencia
MIT
