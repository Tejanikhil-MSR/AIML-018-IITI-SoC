# 🧠 IITI Document QA Pipeline

This project provides an end-to-end pipeline for **scraping**, **preprocessing**, and **querying documents** from the **IIT Indore website** using **Pathway**, **LLMs**, and a **Streamlit-based GUI**.

---

## 📁 Project Structure

```
IITI-Doc-QA/
│
├── data_preparation/         # Scripts for scraping and cleaning data
│   └── placeholder.py        # Entry point for scraping/preprocessing
│
├── data/                     # Directory to store processed document data
│
├── pathway_pipeline/         # Pathway-based retrieval and QA pipeline
│   └── main.py               # Entry point for the backend pipeline
│
├── App/                      # Streamlit GUI app
│   └── app.py                # Streamlit interface
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Quick Start

### ✅ 1. Install All Dependencies

```bash
pip install -r requirements.txt
```

---

### 📥 2. Scrape and Preprocess Data

```bash
python data_preparation/placeholder.py
```

---

### 🧩 3. Run the Pathway QA Pipeline

```bash
python pathway_pipeline/main.py
```

---

### 🧪 4. Test the API via `curl`

```bash
curl -X POST http://localhost:8011      -H "Content-Type: application/json"      -d '{"messages": "Who is the director of IIT Indore?"}'
```

---

## 🖥️ GUI Mode with Streamlit

### 🧩 1. Install Streamlit

```bash
pip install streamlit
```

### 🚀 2. Launch the App

```bash
streamlit run App/app.py
```

Open your browser at [http://localhost:8501](http://localhost:8501)


---

## 📞 Contact

For questions, feature requests, or contributions, feel free to:

- Open a GitHub issue
- Submit a pull request
- Email the project maintainer

---

Happy coding! 🚀
