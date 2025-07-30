# 🧠 IITI Document QA Pipeline

> Throughout the project, LLMs are used for two purposes : `Summarization` and `Coherency`

This project provides an end-to-end pipeline for **scraping**, **preprocessing**, and **querying documents** from the **IIT Indore website** using **Pathway**, **LLMs**, and a **Streamlit-based GUI**.

---

## 📁 Project Structure

```
IITI-DOC-QA/
│
├── app/                      # Flask API and LLM response generation
│   ├── batch_processor.py
│   ├── flask_api_expose_cookies_storage.py
│   └── flask_api_expose_redis_storage.py (optional)
│
├── core/                     # Core backend functionalities
│   ├── loader.py
│   ├── query_classifier.py
│   ├── rag_chain_builder.py
│   └── vector_store_client.py
│
├── pathway_server/           # Pathway-based retrieval and QA pipeline
│   ├── PathwayServerExpose.py
│   └── helpers.py
│
├── Streamlit/                # Streamlit GUI
│   └── app.py
│
├── requirements.lock         # Python dependencies (UV based)
└── README.md                 # Documentation
```

---

## ⚙️ Quick Start

### ✅ 1. Install Dependencies

```bash
uv pip install -r requirements.lock
```

### 🧩 2. Run the Chatbot Backend

```bash
python3 pathway_server/PathwayServerExpose.py
python3 app/flask_api_expose_cookies_storage.py
```

### 🧪 3. API Endpoints

* **GET** `/`

  * Returns model info and a live status message.

* **POST** `/`

  * **Phase-1: Intent Classification**

    * Input: `messages`
    * Output: `status = label_selection_needed`, `probable_labels`
  * **Phase-2: RAG Generation**

    * Input: `messages`, `selected_label`
    * Output: `status = processing_with_label`, `request_id`

* **GET** `/status/<request_id>`

  * Output:

    * `status = processing | completed | error`
    * `message`

---

## 💾 GUI Mode with Streamlit

### 🧩 1. Environment Setup

```bash
pip install streamlit
```

### 🚀 2. Launch the App

```bash
streamlit run Streamlit/app.py
```

---

## 📊 Dataset

* **Source**: [https://iiti.ac.in](https://iiti.ac.in)

* **Formats**: PDFs, Images, Text

* **Collection Phases**:

  * Extracted using Selenium
  * Organized by categories
  * OCR & Summarization using `layout-parser`

* **Cleaning**:

  * Removed non-English text
  * Lowercased
  * Removed repetitions and broken phrases

* **Access**:

  * 👉 [Google Drive](https://drive.google.com/drive/folders/1ubBSaZ34idOf1ZyN_RhGli2YiKKV3Czw)
  * 👉 [Hugging Face](https://huggingface.co/datasets/ankitK-11/iiti-bot-dataset/tree/main)


---

#### 4. RAG Pipeline

- Intent Classification (Phase-1) :
	- Upon receiving a new user query (without a `selected_label`), the `query_classifier` component analyses the query to determine probable intent categories (labels).
	- The application responds by presenting these probable labels to the user, prompting them to select the most relevant one.

- RAG response generation (Phase-2): 
	- When a `selected_label` is provided by the user (indicating they've chosen an intent category), the system proceeds with generating a RAG-based response.
	- The `rag_builder` component is utilized to construct a formatted prompt, incorporating the original user message, conversational history, and the selected label.
	- Uses `Asynchronous` processing with batching - `batch_processor.

- Result Retrieval (Phase-3): 
	- Clients can poll this endpoint using the `request_id` received from the initial `/` POST request.
	- Returns `completed` once the response is ready, along with the generated response.
	- Return `error` if an issue occured during processing.

- Intermediate Global Memory (Sits between the Frontend and the RAG-Backend): (Future Improvements)
	- `Requirement` - Most of the user queries will be of similar type
	- `Objective` - Mechanism that checks if the similar kind `question` is queried in the past and is cached or not. 
	- `Purpose` - Avoid LLM triggers all the time which will reduce the latency of the response. 

---

#### 5. Scaling-up and Optimization

- Memory Management :
	- User chat history (`HumanMessage` and `AIMessage`) is persistently stored within the Flask session (`session['chat_messages']`).
	- The `langchain.memory.ConversationBufferMemory` is used to manage and format the chat history for input to the language model.
	- **`get_user_memory()`:** Retrieves the chat history from the session and populates a `ConversationBufferMemory` object.
	- **`save_user_memory_to_session()`:** Serializes the `ConversationBufferMemory` messages back into a list and saves them to the Flask session after each interaction.

- LLM Inference Optimization : 
	- `Latency` : Uses `KV-Cache` for the response generation for the user query and intent.
	- `Throughput` : Uses batch processing (Assuming that the bot gets multiple requests at a time) -`Not needed for our usecase but implemented`

- Rule-Based Greeting/Send-Off : 
	- Identifies common greeting and send-off keywords in user queries.
	- Provides randomized, predefined responses for such conversational turns, avoiding unnecessary LLM calls.
	- Updates the session memory directly for these immediate responses.

---

## ⚙️ Configurations

* `FLASK_HOST`, `FLASK_PORT`, `FLASK_SECRET_KEY`
* `MODEL_CHOICE`, `EMBEDDING_MODEL`
* `SYSTEM_PROMPT`, `PROMPT_TEMPLATE`
* `MAX_BATCH_SIZE`, `BATCH_TIMEOUT_SECONDS`

---

## 🔗 Concepts Used

* Retrieval Augmented Generation (RAG)
* Real-time ETL streaming
* OCR
* Prompt Engineering
* Caching (KV)
* Batching
* Session Management
* Clustering
* Web scraping
* LLMs

---

#### 8. External Libraries Used

- `asyncio` - For asynchronous processing
- `threading` - For handling batches and multi requests
- `pathway` - For real time streaming
- `Langchain` : Used for creating Pipelines
- `Flask` : Used for creating the endpoints
- `Streamlit` : Used for creating the user interface
- `Selenium` : Scraping the data from Websites
- `HDBScan` : Clustering approach used for document segregation
- `Sentence-Transformer` : Used as an embeddin model in the Document Clustering phase
- `TfidfVectorizer` : For deciding on naming of the labels based on the most frequently used words
- `NLTK` : For removal of stop words from the documents in the Data segregation phase
- `KeyBert` : For keyword and key-phrase extraction from the text-documents using `BERT` 
- `Layout-Parser` : An open-source `deep learning library` for performing OCR on PDFs
- `unstructured` for better extraction of data from PDF documents
- `Prompt Engineering` : For Summarizing the pdfs containing `unstructured data`, `noisy data` and `tabular data`
- `OLLAMA` : Used `LLAMA2` as a Summarization model as it balances both `user_requests` and the `one-shot context` provided 

---

#### 9. Data structures Used :
- `Queues`
- `Lists`
- `Dictionaries`

---


## 📞 Contact

For questions, feature requests, or contributions, feel free to:

- Open a GitHub issue
- Submit a pull request

---

Happy coding! 🚀
