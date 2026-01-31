# 📚 RAG PDF Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that enables users to upload PDF documents and ask questions about their content. Built with **Streamlit**, **Azure OpenAI**, and **FAISS** vector database.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Features

- ✅ **PDF Upload & Processing** - Upload any PDF document for instant analysis
- ✅ **Intelligent Q&A** - Ask questions and get accurate answers with source citations
- ✅ **Anti-Hallucination** - Strict context grounding to prevent made-up information
- ✅ **Source Attribution** - View exact passages used to generate each answer
- ✅ **Adjustable Retrieval** - Control the number of context chunks (3-10)
- ✅ **Chat History** - Maintain conversation context throughout your session
- ✅ **Local Embeddings** - Free HuggingFace embeddings (no API costs)
- ✅ **Clean UI** - Modern, responsive Streamlit interface

## 🏗️ Architecture
```
┌─────────────┐
│   PDF File  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Text Extraction    │
│  (PyPDF)            │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Chunking           │
│  (1000 chars)       │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Embeddings         │
│  (HuggingFace)      │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Vector Store       │
│  (FAISS)            │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐      ┌─────────────────┐
│  Retrieval (Top-K)  │─────▶│  Azure OpenAI   │
└─────────────────────┘      │  (GPT-4)        │
                             └────────┬────────┘
                                      │
                                      ▼
                             ┌─────────────────┐
                             │  Answer + Cites │
                             └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Azure OpenAI API access
- pip package manager

### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/yourusername/rag-pdf-chatbot.git
   cd rag-pdf-chatbot
```

2. **Create virtual environment**
```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
```env
   AZURE_OPENAI_API_KEY=your-api-key-here
   AZURE_ENDPOINT=https://your-resource.openai.azure.com/
   OPENAI_API_VERSION=2024-02-15-preview
   AZURE_DEPLOYMENT_NAME=gpt-4o
```

5. **Run the application**
```bash
   streamlit run app.py
```

6. **Open your browser**
   
   Navigate to `http://localhost:8501`

## 📦 Project Structure
```
rag-pdf-chatbot/
│
├── app.py                  # Main Streamlit application
├── .env                    # Environment variables (not in repo)
├── .env.example           # Example environment file
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore rules
│
└── docs/                 # Documentation (optional)
    ├── architecture.md   # Detailed architecture
    └── setup-guide.md   # Detailed setup instructions
```

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit | Web interface |
| **LLM** | Azure OpenAI (GPT-4) | Answer generation |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | Text vectorization |
| **Vector DB** | FAISS | Similarity search |
| **PDF Processing** | PyPDF | Document parsing |
| **Framework** | LangChain | RAG orchestration |

## 💡 Usage

1. **Upload a PDF**
   - Click "Choose a PDF file" in the sidebar
   - Select your document

2. **Process the Document**
   - Click "🔄 Process PDF"
   - Wait for chunking and embedding generation

3. **Ask Questions**
   - Type your question in the chat input
   - View answer with source citations

4. **Adjust Settings (Optional)**
   - Use the slider to control chunk retrieval (3-10)
   - Higher numbers = more context but slower

5. **Clear Chat**
   - Click "🗑️ Clear Chat" to start fresh

## ⚙️ Configuration

### Chunk Size & Overlap

Adjust in `app.py` (lines 81-82):
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200     # Overlap between chunks
)
```

### Retrieval Settings

Default retrieves 5 chunks. Adjust via slider in UI or in code (line 156):
```python
search_kwargs={"k": 5}  # Number of chunks to retrieve
```

### Temperature

Set to 0 for factual responses (line 133):
```python
temperature=0  # 0 = deterministic, 1 = creative
```

## 🔒 Anti-Hallucination Features

1. **Strict Prompting** - Explicit instructions to only use provided context
2. **Source Markers** - Each chunk labeled with source and page number
3. **Temperature 0** - Deterministic responses
4. **Full Source Display** - Complete text shown for verification
5. **System Prompt** - "Never make up information" directive

## 📊 Example Queries
```
✅ "What is the main topic of this document?"
✅ "Summarize the key findings from page 5"
✅ "What does the document say about [specific topic]?"
✅ "List all requirements mentioned in section 3"
✅ "What are the conclusions?"
```

## 🐛 Troubleshooting

### Issue: "Module not found" error
**Solution:** Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: Azure OpenAI 404 error
**Solution:** Check your `.env` file:
- Verify `AZURE_ENDPOINT` is correct
- Ensure `AZURE_DEPLOYMENT_NAME` matches your Azure deployment
- Confirm API key is valid

### Issue: PDF not loading
**Solution:** 
- Ensure PDF is not password-protected
- Check file size (very large PDFs may take time)
- Verify PDF is text-based, not scanned images

### Issue: Slow embedding generation
**Solution:**
- First run downloads the embedding model (~90MB)
- Subsequent runs use cached model
- Consider smaller PDFs for testing

## 🚧 Roadmap

- [ ] Support for multiple PDFs simultaneously
- [ ] Conversation memory across sessions
- [ ] Export chat history to PDF
- [ ] Support for DOCX, TXT files
- [ ] Multi-language support
- [ ] OCR for scanned PDFs
- [ ] Deploy to cloud (Azure/AWS/Heroku)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for RAG framework
- [Streamlit](https://streamlit.io/) for the amazing UI framework
- [HuggingFace](https://huggingface.co/) for free embedding models
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - your.email@example.com

Project Link: [https://github.com/sathishbabu89/langchain-rag-pdf-chatbot]([https://github.com/yourusername/rag-pdf-chatbot](https://github.com/sathishbabu89/langchain-rag-pdf-chatbot))

---

⭐ **If you found this helpful, please give it a star!** ⭐
