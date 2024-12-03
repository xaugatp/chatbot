# **Allianz Car Insurance Chatbot**

## **Overview**
The Allianz Car Insurance Chatbot is a state-of-the-art solution designed to process large volumes of PDF documents and provide users with concise, accurate answers to their queries. This chatbot leverages advanced Natural Language Processing (NLP) models and an intuitive chat interface to deliver real-time, context-aware responses.

This project integrates tools such as LangChain, HuggingFace models, and ChromaDB to provide a seamless and efficient document query system. The application is hosted via Solara, ensuring a user-friendly web interface for interaction.

---

## **Key Features**
1. **PDF Merging and Processing**:
   - Automatically merges multiple PDF documents from a specified directory.
   - Processes the merged document and splits it into manageable text chunks for better querying.

2. **Vector Database**:
   - Uses ChromaDB for efficient document indexing and retrieval.
   - Embeds document chunks using HuggingFace’s sentence-transformers for high-quality semantic search.

3. **AI-Powered Responses**:
   - Utilizes HuggingFace’s pre-trained Falcon-7B-Instruct model for generating context-aware responses.
   - Provides concise, clear, and accurate answers based on the provided document context.

4. **Interactive Chat Interface**:
   - Real-time chat interface built with Solara.
   - Maintains a user-friendly environment for query submission and response visualization.

5. **Error Handling**:
   - Gracefully manages errors during PDF processing, vector store initialization, and response generation.

---

## **Project Structure**
```
Allianz-Car-Insurance-Chatbot/

│   ├── sol.py             # Main Solara-based application logic
│   ├── config.json        # Configuration file for specifying input directory
│   └── requirements.txt   # Python dependencies
│   ├── merged_document.pdf # Output of the PDF merging process
│   ├── chroma_db/         # Directory for ChromaDB persistence
├── README.md              # Project documentation
```

---

## **Getting Started**

### **Prerequisites**
1. **Python**: Ensure Python 3.8 or later is installed.
2. **Environment**: Set up a virtual environment (recommended for managing dependencies).
3. **API Key**: A HuggingFace API token is required for accessing models.

### **Installation**
1. Clone the repository or download the zip file. 
   ```bash
   git clone https://github.com/xaugatp/chatbot.git
   cd Allianz-Car-Insurance-Chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Update the `config.json` file:
   - Set the `pdf_directory` field to the directory containing your PDF documents.

---

### **Usage**

1. **Merge PDF Files**:
   - Place the PDF files in the directory specified in `config.json`.
   - The script automatically merges the PDFs and processes them for querying.

2. **Run the Chatbot**:
   ```bash
   solara run sol.py
   ```
   - This starts the Solara-based chatbot interface.
   - Open the provided URL (usually `http://localhost:8765`) in your browser.

3. **Interact**:
   - Type your queries in the chatbox.
   - Receive AI-generated responses based on the uploaded documents.

---

## **Technical Details**

### **PDF Processing**
- The chatbot merges all PDFs in the specified directory using `PyPDF2`.
- Documents are chunked into 500-character segments with 150-character overlap for optimal processing.

### **Embeddings and Vector Store**
- HuggingFace’s `sentence-transformers/all-MiniLM-L6-v2` model generates embeddings.
- ChromaDB stores these embeddings and retrieves the most relevant chunks for each query.

### **Language Model**
- HuggingFace's Falcon-7B-Instruct model powers the response generation.
- The model is fine-tuned for concise, context-aware question answering.

### **Frontend**
- Built with Solara, the chatbot provides a responsive and interactive user interface.

---

## **Customization**

### **Change PDF Directory**
Update the `config.json` file with your desired directory:
```json
{
    "pdf_directory": "path/to/your/pdf/directory"
}
```

### **Modify Model Parameters**
In `sol.py`, adjust the HuggingFaceHub model parameters to suit your use case:
```python
llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.5, "max_new_tokens": 250})
```

---

## **Troubleshooting**

### **Common Errors**
1. **PDF Directory Not Found**:
   - Ensure the `pdf_directory` path in `config.json` is correct and accessible.

2. **HuggingFace API Issues**:
   - Check your HuggingFace API token and ensure it’s correctly set in your environment.

3. **ChromaDB Errors**:
   - Delete the `chroma_db` directory if you encounter initialization errors and let the app regenerate it.

---


## **License**
This project is licensed under the [MIT License](LICENSE).

---

## **Acknowledgements**
- [HuggingFace](https://huggingface.co/) for providing world-class NLP models.
- [LangChain](https://langchain.com/) for robust document processing tools.
- [Solara](https://solara.dev/) for an elegant web application framework. 

---

For questions, suggestions, or support, please open an issue or contact me at saugatp363@gmail.com*
