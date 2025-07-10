

> [!NOTE] Note
> Throughout the project, LLMs are used for two purposes : `Summarization` and `Coherency`


#### File structure

- `App.py` - Endpoint that deals with request and response handling
- `batch_processor.py` - Contains the `BatchProcessor` class, which is responsible for efficiently managing and executing requests to the underlying Language Model (LLM).
- `loader.py` - Loads the generation LLM model, tokenizers, also handles batched response generation
- `PathwayServerExpose.py` - Deals with real time monitoring and Lazy transformations and exposing the vector store to the other processess
- `rag_chain_builder.py` - Deals with retrieval of documents using the `user prompt` + `label`, send it for augmentation using generation LLM
- `query_classifier.py` - Predicts the user intents based on the query provided
- `config.py` - Defines all the configurations for the project
- `logs/` - Records all the pathway server logs


#### How to interact ? (API Endpoints)

- **GET** `/` 
	- Returns a simple string indicating the chatbot is live and the model in use.
- **POST** `/`
	- Initial Query (Phase-1: Intent Classification)
		- **Request Body** : (`JSON`)
			- Fields : `messages`
		- **Response** : (`JSON`)
			- Fields : `status = label_selection_needed`, `message`, `query`, `probable_labels`
	- Labeled query (Phase-2 : RAG Generation)
		- **Request Body** : (`JSON`)
			- Fields : `messages`, `selected_label`
		- **Response** : (`JSON`)
			- Fields : `request_id`, `status = processing_with_label`, `label`
- **GET** `/status/<request_id>`
	- **Response (Processing)**
		- Fields : `status = processing`, `message = Processing...`
	- **Response (Completed)**
		- Fields : `status = completed`, `message`
	- **Response (Error)** 
		- Fields = `status = error`, `message`

#### Dataset Collection

- Source : College Website (`iiti.ac.in`)
- Types of documents considered so far : `PDFs`, `Text`
- Collection and Organization of data is done in multiple phases : 
	- **Extraction** from the IIT Indore website using `Selenium` library and the extracted data is organized as follows : 
		- `Folders` named after the category mentioned in the website
		- Each folder has a `.txt` file, `Images/` folder containing all the images, `Downloaded/` folder containing the downloaded `PDF` files
	- **Manual** go through 
		- Observed that the data has many `broken phrases`, `word repetations`, `multi-lingual`, `empty documents`
	- **Pre-processing** of data 
		- Removal of `non-english` phrases.
		- `lower-casing` the words to reduce the size of vocabulary
		- Performing `OCR` on PDF documents using `layout-parser` followed by summarization of documents.
	- **Data Organization** : 
		- Clustered all the documents by capturing the contextual information and organized them with corresponding `subfolders` named after the `intent/category`


#### RAG Pipeline

- Intent Classification (Phase-1) :
	- Upon receiving a new user query (without a `selected_label`), the `query_classifier` component analyses the query to determine probable intent categories (labels).
	- The application responds by presenting these probable labels to the user, prompting them to select the most relevant one.
	- The original user message is temporarily stored in the session (`session['original_user_message']`) for later use.

- RAG response generation (Phase-2): 
	- When a `selected_label` is provided by the user (indicating they've chosen an intent category), the system proceeds with generating a RAG-based response.
	- The `rag_builder` component is utilized to construct a formatted prompt, incorporating the original user message, conversational history, and the selected label.
	- Uses `Asynchronous` processing with batching - `batch_processor.

- Result Retrieval (Phase-3): 
	- Clients can poll this endpoint using the `request_id` received from the initial `/` POST request.
	- Returns `completed` once the response is ready, along with the generated response.
	- Return `error` if an issue occured during processing.
	- Upon successful completion, the `updated_chat_messages` from the RAG process are saved back into the Flask `session['chat_messages']`, ensuring conversational continuity.

- Intermediate Global Memory (Sits between the Frontend and the RAG-Backend):
	- `Requirement` - Most of the user queries will be of similar type
	- `Objective` - Mechanism that checks if the similar kind `question` is queried in the past and is cached or not. 
	- `Purpose` - Avoid LLM triggers all the time which will reduce the latency of the response. 

#### Scaling-up and Optimization

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

#### Important Configurations

- `FLASK_HOST`
- `FLASK_PORT`
- `FLASK_SECRET_KEY`
- `MODEL_CHOICE` - Response generation LLM
- `EMBEDDING_MODEL` - Creating Vector Stores
- `SYSTEM_PROMPT`
- `PROMPT TEMPLATE`
- `MAX_BATCH_SIZE`
- `BATCH_TIMEOUT_SECONDS`

#### Concepts Used

- Retrieval Augmented Generation (`RAG`)
- Real-time streaming (`ETL`)
- Optical character recognition (`OCR`)
- Prompt Engineering (Designing)
- KV-Cache
- Batching
- Session Management
- Web-scraping
- Clustering
- Large-Language Models (`LLM`)

#### External Libraries Used

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
- `Prompt Engineering` : For Summarizing the pdfs containing `unstructured data`, `noisy data` and `tabular data`
- `OLLAMA` : Used `LLAMA2` as a Summarization model as it balances both `user_requests` and the `one-shot context` provided 

#### Data structures Used :
- `Queues`
- `Lists`
- `Dictionaries`

#### Whats Left ? (Deadline : Sunday 13/06/2025 11:59 PM)

- Updation of RAG knowledge Base based on the dynamic web-content (Assigned to @Saransh, @Ayush)
- Ensuring that the data being updated is being processed by the pathway properly (Assigned to @Amir)
- Organization of Data
- Prettify Frontend (Assigned to @Tejanikhil)
- Intermediate Memory integration (Assigned to @JatinSharma)
- RAG Evaluation (Assigned to @Ankit)