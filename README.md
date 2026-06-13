# MultiModal-RAG

**MultiModal-RAG** is an AI-powered conversational assistant that searches a photo database using natural language queries. It retrieves the most relevant image along with its captions by mapping both into the same CLIP vector space, and then uses a Large Language Model via Hugging Face to answer any questions about the retrieved image.

---

## Features

- **Multimodal Retrieval:** Embeds text queries, images, and captions into a shared vector space using OpenCLIP and ChromaDB.
- **Conversational Interface:** A clean, visually appealing chat UI built with Streamlit.
- **Data Ingestion Script:** Easily build the vector database from your local images and text captions using `ingest.py`.
- **Model Selection:** Easily swap out the underlying model by changing the Hugging Face repository ID directly in the UI.
- **Powered by LLaMA 3:** Uses `meta-llama/Meta-Llama-3-8B-Instruct` via Hugging Face Inference API by default.

---

## Data Setup

Before running the application, you must set up the data and build the vector database:

1. Ensure there is a `data` folder in the root directory.
2. Inside `data/`, create an `images/` directory and place your `.jpg` files inside (e.g., from the Flickr30k dataset).
3. Add a `captions.txt` file inside `data/` with the columns: `image_name, comment_number, comment`.
4. Once the data is in place, run the ingestion script to embed everything into ChromaDB:
   ```bash
   python ingest.py
   ```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/MultiModal-RAG.git
   cd MultiModal-RAG
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv env
   env\Scripts\activate        # Windows
   # source env/bin/activate   # Linux / macOS
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file** inside the `MultiModal-RAG` folder and add your Hugging Face API token using this exact template:
   ```env
   HUGGINGFACEHUB_API_TOKEN = "your_API_Key"
   ```
   > Get your free API token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).  

5. **Run Data Ingestion:**
   ```bash
   python ingest.py
   ```

6. **Run the app:**
   ```bash
   streamlit run app.py
   ```

---

## How to Use

1. Open the app in your browser (Streamlit will provide a local URL).
2. (Optional) Configure the assistant in the sidebar:
   - **Model (HF repo ID):** Change to any text-generation model.
3. Type a query (e.g. "a dog playing in the snow") into the search bar and click **Search**.
4. The system will retrieve the most relevant image and top 4 captions from the database.
5. Type your message in the chat input to ask the AI questions about the retrieved image.
6. Use the **Clear conversation** button in the sidebar to start fresh.

---

## How to Use a Custom Hugging Face Model

1. Create a `.env` file in the project root:
   ```env
   HUGGINGFACEHUB_API_TOKEN=your_hugging_face_token_here
   ```

2. Change the model directly in the app's sidebar under **HF Model Repo**.

3. Any model available on the [Hugging Face Hub](https://huggingface.co/models) that supports the Inference API can be used.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| Vector Database | ChromaDB |
| Embeddings | OpenCLIP |
| LLM Orchestration | LangChain |
| Model Provider | Hugging Face Inference API |
| Environment Config | python-dotenv |

---

## License

[MIT](LICENSE)
