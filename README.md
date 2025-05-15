# Multimodal RAG System with PDF and Image Processing

## Description

This Jupyter Notebook implements a comprehensive Multimodal Retrieval Augmented Generation (RAG) system. It processes PDF documents to extract both text and images, generates embeddings for this content, and uses these embeddings to create searchable vector stores. The system then integrates a Large Language Model (LLM), specifically `mistralai/Mistral-7B-Instruct-v0.1`, to answer questions based on the retrieved multimodal context. The notebook also includes phases for evaluating the RAG system and visualizing the generated embeddings. Finally, it sets up a Streamlit application to provide a user interface for interacting with the RAG system.

## Prerequisites and Installation

Before running the notebook, ensure you have the necessary dependencies installed. The notebook includes cells to install the following:

* **Langchain and related libraries:**
    ```bash
    pip install langchain langchain-community langchain-huggingface langchain-text-splitters pypdf pymupdf
    ```
* **Vector store and embedding libraries:**
    ```bash
    pip install faiss-cpu sentence-transformers transformers torch torchvision Pillow
    ```
* **OCR and PDF/Image handling:**
    ```bash
    pip install pytesseract pdf2image
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr libtesseract-dev poppler-utils
    ```
* **LLM and Hugging Face libraries:**
    ```bash
    pip install -q transformers accelerate bitsandbytes huggingface_hub --quiet
    ```
    * **Important:** After running this installation, you might need to **RESTART THE RUNTIME** for the changes to take effect.
* **Evaluation and Visualization libraries:**
    ```bash
    pip install nltk rouge-score scikit-learn matplotlib seaborn --quiet
    ```
    NLTK data (punkt, punkt_tab) will be downloaded if not present.
* **Streamlit and ngrok (for UI):**
    ```bash
    pip install streamlit pyngrok --quiet
    ```

You will also need to:
* Mount your Google Drive to access PDF files if they are stored there. The notebook provides cells for this:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
* Set up your Hugging Face token. The notebook attempts to get it from Colab secrets or prompts for a manual login:
    ```python
    from google.colab import userdata
    hf_token = userdata.get('HF_TOKEN')
    # ... (manual login if not found)
    ```
* Configure the `pdf_directory` variable to point to the folder containing your PDF files:
    ```python
    pdf_directory = '/content/drive/MyDrive/Semester_8/GenAi/A3/Data' # *** Replace with your PDF folder path ***
    ```

## Workflow / Phases

The notebook is structured into several phases:

1.  **Setup and Environment Check:**
    * Imports necessary libraries.
    * Checks for CUDA availability and sets the device (GPU or CPU).
    * Mounts Google Drive.
    * Installs required packages for LangChain, PDF processing, OCR, embeddings, and the LLM.

2.  **Data Loading and Preprocessing (PDFs and Images):**
    * Loads text content from PDF documents using `PyMuPDFLoader`.
    * Extracts images from PDF documents using `PyMuPDF (fitz)`.
    * Performs OCR on extracted images using `pytesseract` to get textual content from images.
    * Creates LangChain `Document` objects for both text pages and extracted images, storing relevant metadata (source PDF, page number, image path, OCR text).

3.  **Text Splitting:**
    * Splits the extracted text documents into smaller chunks using `RecursiveCharacterTextSplitter` for more effective embedding and retrieval.

4.  **Embedding Model Initialization:**
    * Initializes a text embedding model (`sentence-transformers/all-MiniLM-L6-v2`) using `HuggingFaceEmbeddings` from LangChain.
    * Initializes an image embedding model (CLIP: `openai/clip-vit-base-patch32`) directly using the `transformers` library.
    * Defines a helper function `embed_image_clip` to generate embeddings for images.

5.  **Vector Store Creation:**
    * Creates a FAISS vector store for the text chunks using the text embedder.
    * Generates embeddings for the image documents using the CLIP image embedder.
    * Creates a separate FAISS vector store for the image embeddings manually and saves it along with a document map.
    * Saves the text vector store locally (`faiss_text_index`).
    * Saves the image FAISS index (`faiss_image_index.faiss`) and document map (`faiss_image_doc_map.pkl`).

6.  **Language Model (LLM) Integration:**
    * Handles Hugging Face token authentication.
    * Initializes the LLM (`mistralai/Mistral-7B-Instruct-v0.1`) and its tokenizer using `BitsAndBytesConfig` for 4-bit quantization to save memory.
    * Defines a Mistral-specific prompt template for RAG.
    * **Note:** There's an error message in the notebook output indicating `bitsandbytes` library might not have been found or the runtime wasn't restarted after installation, which prevented 4-bit loading. This needs to be resolved for the LLM to load correctly.

7.  **Semantic Search and Retrieval:**
    * Loads the previously saved FAISS indices for text and images.
    * Defines helper functions for:
        * Embedding text queries using CLIP's text encoder (`embed_text_clip`).
        * Retrieving relevant text chunks (`retrieve_text`).
        * Retrieving relevant images based on a text query (`retrieve_images`).
        * Combining text and image retrieval results (`multimodal_retrieve`).
    * Runs example retrieval queries.

8.  **RAG Chain Construction and Execution:**
    * Defines a function `format_context` to prepare retrieved documents for the LLM prompt.
    * Defines a custom inference function `run_mistral_inference` to generate responses from the LLM.
    * Builds a RAG chain using LangChain Expression Language (LCEL), combining retrieval, context formatting, prompt templating, and LLM inference.
    * Executes the RAG chain with example queries.
    * **Note:** The notebook output indicates the RAG chain could not be built due to the LLM/Tokenizer not being initialized (likely related to the `bitsandbytes` issue mentioned in Phase 6).

9.  **Evaluation Setup:**
    * Installs libraries for evaluation (NLTK, ROUGE, scikit-learn, etc.).
    * Defines ground truth data (manually created examples of queries, relevant document IDs, and reference answers).
    * Defines functions to:
        * Get unique document identifiers (`get_doc_id`).
        * Calculate Precision@K and Recall@K for retrieval (`calculate_precision_recall_at_k`).
        * Calculate BLEU and ROUGE scores for generation quality (`calculate_bleu`, `calculate_rouge`).
    * **Note:** The ground truth data in the notebook is illustrative and needs to be populated with actual IDs and relevant content from the processed documents.

10. **Run Evaluation:**
    * Iterates through the ground truth queries.
    * For each query:
        * Retrieves documents using `multimodal_retrieve`.
        * Generates an answer using the RAG chain (or the LLM directly with retrieved context).
        * Calculates latency, retrieval metrics (Precision@K, Recall@K), and generation metrics (BLEU, ROUGE).
    * Calculates and prints average metrics.
    * **Note:** The evaluation loop also encountered an error due to the `rag_chain` not being found (linked to the LLM initialization issue).

11. **Embedding Visualization:**
    * Prepares text and image embeddings for visualization.
    * Defines a function `run_tsne_and_plot` to perform t-SNE dimensionality reduction on embeddings and plot them using `matplotlib` and `seaborn`.
    * Visualizes text embeddings (SentenceTransformer) and image embeddings (CLIP) separately.

12. **Streamlit UI Application (app.py):**
    * Installs `streamlit` and `pyngrok`.
    * Writes a Python script `app.py` for the Streamlit application.
    * The `app.py` script includes:
        * A cached function `load_all_resources` to load all models, tokenizers, and vector stores.
        * Helper functions for embedding, retrieval, context formatting, and LLM inference (copied or adapted from the notebook).
        * The main RAG response function `get_rag_response` that orchestrates the retrieval and generation.
        * Streamlit UI elements for user input (text query, optional image upload) and displaying the answer and sources.
    * Authenticates `ngrok` with an auth token.
    * Runs the Streamlit app and creates a public URL using `pyngrok` to make the UI accessible.

## Key Components

* **PDF Processing:** Extracts text and images from PDFs.
* **OCR:** Converts images (especially those from PDFs) into text.
* **Text and Image Embeddings:** Uses SentenceTransformers for text and CLIP for images.
* **Vector Stores:** FAISS is used for efficient similarity search of text and image embeddings.
* **Language Model:** Mistral-7B-Instruct-v0.1 for generation.
* **RAG Pipeline:** Combines retrieval of relevant multimodal context with LLM generation.
* **Evaluation:** Metrics like Precision, Recall, BLEU, and ROUGE.
* **Visualization:** t-SNE plots for embeddings.
* **User Interface:** Streamlit app for interaction.

## Configuration and Usage Notes

* **File Paths:** Ensure `pdf_directory`, `faiss_image_index_path`, `image_doc_map_path`, and `text_vector_store_path` are correctly set to your environment. The notebook saves FAISS indices locally in the Colab environment.
* **Hugging Face Token:** Required for downloading models from Hugging Face Hub.
* **Runtime Restart:** After installing `bitsandbytes` and `accelerate`, a runtime restart in Colab is crucial for the 4-bit quantization of the LLM to work. The notebook output shows errors related to this, suggesting this step might have been missed or failed.
* **Ground Truth for Evaluation:** The `ground_truth_data` dictionary in Phase 6 needs to be carefully populated with actual document/chunk identifiers and relevant reference answers for meaningful evaluation.
* **Streamlit App (`app.py`):**
    * The `app.py` script is designed to be run in an environment where the FAISS index files (`faiss_text_index/`, `faiss_image_index.faiss`, `faiss_image_doc_map.pkl`) are accessible in the current working directory.
    * The ngrok auth token needs to be correctly set for the public URL to work.
    * The LLM loading in `app.py` (within `load_all_resources`) also depends on `bitsandbytes` and a compatible environment.

## Running the Streamlit App

1.  Ensure all previous notebook cells have been run successfully, especially those that save the FAISS indices and the LLM model components (if not downloaded on the fly by `app.py`).
2.  Make sure `app.py` has been written to the Colab environment (Phase 5, Step 3).
3.  Authenticate ngrok with your token (Phase 5, Step 4).
4.  Run the cell that starts the Streamlit app using `subprocess` and `pyngrok` (Phase 5, Step 5).
5.  Access the provided ngrok public URL in your browser.
