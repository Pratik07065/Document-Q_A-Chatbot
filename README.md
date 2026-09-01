# Document RAG Chatbot 📄

A Streamlit-based web application that enables intelligent question-answering over PDF and Word documents using Retrieval Augmented Generation (RAG) powered by Google Gemini API.

## Features

- **Multi-Format Support**: Upload and process both PDF and DOCX (Word) files
- **Dual Embedding Options**: 
  - Local embeddings using Sentence Transformers (no API quota usage)
  - Cloud embeddings using Google Gemini API
- **Vector Search**: FAISS-based similarity search for fast document retrieval
- **RAG Pipeline**: LangChain-powered retrieval-augmented generation for accurate answers
- **Chat Interface**: Interactive Streamlit UI with conversation history
- **Hallucination Prevention**: Strict prompting to ensure answers come only from provided documents

## Project Structure

```
RAG-CHATBOT/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Prerequisites

- Python 3.8 or higher
- Google Gemini API Key (for cloud embedding or GPT integration)
- At least 2GB RAM for local embeddings

## Installation

1. **Clone or download the project** to your local machine

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### Getting a Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your API key and paste it in the app sidebar

## Usage

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **In the sidebar**:
   - Enter your Google Gemini API Key
   - Choose embedding method:
     - **Local Embeddings**: Runs on your machine (recommended for privacy)
     - **Gemini Cloud API**: Uses Google's cloud service
   - Select the embedding model
   - Upload your PDF or DOCX files

3. **Process documents**: Click the "Process Documents" button

4. **Ask questions**: Type your question in the chat input and get answers based on your documents

## How It Works

1. **Document Extraction**: Text is extracted from uploaded PDFs and DOCX files
2. **Chunking**: Large documents are split into overlapping chunks for better retrieval
3. **Embedding**: Text chunks are converted to embeddings using your chosen method
4. **Vector Storage**: Embeddings are stored in FAISS for fast similarity search
5. **Retrieval**: User questions are embedded and matched against document chunks
6. **Generation**: Retrieved context is passed to Gemini for generating accurate answers

## Supported Embedding Models

### Local Embeddings
- **all-MiniLM-L6-v2**: Lightweight, high-performance model (recommended)
- **BAAI/bge-small-en-v1.5**: Alternative small embedding model

### Cloud Embeddings
- **models/text-embedding-004**: Official Google Gemini embedding model

## Dependencies

- **streamlit**: Web UI framework
- **langchain**: LLM orchestration library
- **langchain-google-genai**: Google Gemini integration
- **faiss-cpu**: Vector similarity search
- **sentence-transformers**: Local embedding models
- **pypdf**: PDF text extraction
- **python-docx**: Word document processing

## Performance Tips

- **Local Embeddings** are faster and don't consume API quotas
- Use **all-MiniLM-L6-v2** for best speed-to-quality ratio
- Process documents with fewer pages for quicker indexing
- The app caches the embedding model for repeated use

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No readable text found" | Ensure PDFs are not scanned images; use text-based PDFs |
| API key errors | Verify your Gemini API key is valid and active |
| Slow document processing | Try reducing file size or using local embeddings |
| Out of memory | Process fewer documents at once or use a machine with more RAM |

## Limitations

- Answers are limited to information in uploaded documents
- Large documents may take longer to process
- Local embeddings work best with English text
- API-based embeddings incur costs based on Google's pricing

## Future Enhancements

- Support for additional file formats (CSV, TXT, PPT)
- Document summarization
- Multi-language support
- Persistent vector store storage
- Export conversation history

## License

This project is open source and available for educational and commercial use.

## Support

For issues, questions, or contributions, please open an issue in the project repository.

---

**Built with using Streamlit, LangChain, and Google Gemini API**
