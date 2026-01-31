# filename: app.py - Using Native OpenAI Client (BEST APPROACH) - Enhanced Version

import streamlit as st
from openai import AzureOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os
import tempfile

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG PDF Chat Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .source-content {
        background-color: #f8f9fa;
        padding: 10px;
        border-left: 3px solid #1E88E5;
        margin: 10px 0;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

@st.cache_resource
def load_embeddings():
    """Load embedding model"""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def get_azure_client():
    """Get Azure OpenAI client - Using native openai library"""
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("OPENAI_API_VERSION"),
    )

def process_pdf(uploaded_file):
    """Process PDF and create vector store"""
    
    with st.spinner("📄 Reading PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        st.info(f"✅ Loaded {len(documents)} pages")
    
    with st.spinner("✂️ Splitting document..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        st.info(f"✅ Created {len(chunks)} chunks")
    
    with st.spinner("🧠 Creating vector store..."):
        embeddings = load_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.success("✅ Vector store ready!")
    
    os.unlink(tmp_file_path)
    return vectorstore

def ask_question(question, retriever):
    """Ask question using Azure OpenAI directly - Enhanced with better grounding"""
    
    # Get Azure client
    client = get_azure_client()
    deployment = os.getenv("AZURE_DEPLOYMENT_NAME")
    
    # Get relevant documents from vector store
    docs = retriever.invoke(question)
    
    # Format context with clear source markers
    context_parts = []
    for i, doc in enumerate(docs, 1):
        page_num = doc.metadata.get('page', 'Unknown')
        context_parts.append(f"[Source {i} - Page {page_num}]\n{doc.page_content}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Enhanced prompt with stricter grounding instructions
    prompt = f"""You are a helpful assistant that answers questions based STRICTLY on the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer ONLY using information from the context above
2. If the answer is in the context, provide a detailed response
3. If the information is NOT in the context, clearly state "I cannot find this information in the provided document"
4. Do NOT make up or infer information that is not explicitly stated
5. When possible, reference which source number supports your answer

ANSWER:"""
    
    # Call Azure OpenAI
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a precise assistant that only answers based on provided context. Never make up information."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    answer = response.choices[0].message.content
    
    return answer, docs

# Main UI
st.markdown('<p class="main-header">📚 RAG PDF Chat Assistant</p>', unsafe_allow_html=True)
st.markdown("Upload a PDF and ask questions about it!")

# Sidebar
with st.sidebar:
    st.header("📁 Upload Document")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
    if uploaded_file is not None:
        st.success(f"✅ {uploaded_file.name}")
        
        if st.button("🔄 Process PDF", type="primary", use_container_width=True):
            try:
                st.session_state.vectorstore = process_pdf(uploaded_file)
                # Retrieve top 5 chunks instead of 3 for better coverage
                st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                )
                st.session_state.pdf_processed = True
                st.session_state.chat_history = []
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    st.header("⚙️ Settings")
    
    if st.session_state.pdf_processed:
        st.success("✅ PDF Ready")
        
        # Add retrieval settings
        num_chunks = st.slider("Chunks to retrieve", 3, 10, 5)
        if st.session_state.retriever:
            st.session_state.retriever.search_kwargs["k"] = num_chunks
    else:
        st.warning("⚠️ Upload PDF first")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    
    # Info
    st.header("ℹ️ About")
    st.markdown("""
    **Tech Stack:**
    - 🧠 Embeddings: HuggingFace
    - 🤖 LLM: Azure OpenAI (Direct)
    - 💾 Vector DB: FAISS
    
    **Anti-Hallucination Features:**
    - ✅ Strict context-only responses
    - ✅ Temperature set to 0
    - ✅ Explicit source attribution
    - ✅ Full source text display
    """)

# Chat interface
if st.session_state.pdf_processed:
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
            if 'sources' in msg:
                with st.expander("📚 View Full Sources", expanded=False):
                    for i, doc in enumerate(msg['sources'], 1):
                        st.markdown(f"### Source {i} - Page {doc.metadata.get('page', '?')}")
                        st.markdown(f'<div class="source-content">{doc.page_content}</div>', unsafe_allow_html=True)
                        st.divider()
    
    # Chat input
    if question := st.chat_input("Ask about your PDF..."):
        
        # User message
        st.session_state.chat_history.append({'role': 'user', 'content': question})
        with st.chat_message('user'):
            st.markdown(question)
        
        # Get answer
        with st.spinner("🤔 Thinking..."):
            try:
                answer, sources = ask_question(question, st.session_state.retriever)
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': answer,
                    'sources': sources
                })
                
                with st.chat_message('assistant'):
                    st.markdown(answer)
                    with st.expander("📚 View Full Sources", expanded=False):
                        for i, doc in enumerate(sources, 1):
                            st.markdown(f"### Source {i} - Page {doc.metadata.get('page', '?')}")
                            st.markdown(f'<div class="source-content">{doc.page_content}</div>', unsafe_allow_html=True)
                            st.divider()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.error(f"Full error details: {type(e).__name__}")
else:
    # Welcome screen
    st.info("👆 Upload a PDF to get started!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📤 Step 1")
        st.markdown("Upload your PDF")
    
    with col2:
        st.markdown("### 🔄 Step 2")
        st.markdown("Process the document")
    
    with col3:
        st.markdown("### 💬 Step 3")
        st.markdown("Ask questions!")
