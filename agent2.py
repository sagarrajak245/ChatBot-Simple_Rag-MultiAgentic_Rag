import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional
import hashlib

# LangChain components
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# LangGraph
from langgraph.graph import StateGraph, START, END

# ================================
# Load environment variables
# ================================
load_dotenv()

# ================================
# Enhanced State Management
# ================================
class AppState(TypedDict, total=False):
    question: str
    context: str
    draft_answer: str
    final_answer: str
    retrieved_docs: List[Dict[str, Any]]
    document_sources: List[str]
    confidence_score: float
    processing_steps: List[str]

# ================================
# Document Processing Utilities
# ================================
class DocumentProcessor:
    """Handles multiple document types and processing"""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.csv': CSVLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.doc': UnstructuredWordDocumentLoader
    }
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def load_document(self, file_path: str, file_name: str) -> List[Dict]:
        """Load a single document and return processed chunks"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        loader_class = self.SUPPORTED_EXTENSIONS[file_ext]
        
        # Special handling for CSV files
        if file_ext == '.csv':
            loader = loader_class(file_path)
        else:
            loader = loader_class(file_path)
        
        docs = loader.load()
        
        # Add metadata about source document
        for doc in docs:
            doc.metadata['source_file'] = file_name
            doc.metadata['file_type'] = file_ext
            doc.metadata['doc_id'] = hashlib.md5(file_name.encode()).hexdigest()[:8]
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(docs)
        
        return chunks
    
    def process_multiple_files(self, uploaded_files) -> tuple[List[Dict], List[str]]:
        """Process multiple uploaded files"""
        all_chunks = []
        processed_files = []
        
        for uploaded_file in uploaded_files:
            try:
                # Save temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name
                
                # Process document
                chunks = self.load_document(temp_path, uploaded_file.name)
                all_chunks.extend(chunks)
                processed_files.append(uploaded_file.name)
                
                # Cleanup
                os.unlink(temp_path)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
        
        return all_chunks, processed_files

# ================================
# Enhanced Chain Components
# ================================
def create_retrieval_chain(vectorstore: FAISS):
    """Create a retrieval chain that finds relevant documents"""
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 6,  # Get more documents for better context
            "score_threshold": 0.1
        }
    )
    
    def format_docs(docs):
        """Format retrieved documents with source information"""
        formatted = []
        for doc in docs:
            source_info = f"[Source: {doc.metadata.get('source_file', 'Unknown')}]"
            formatted.append(f"{source_info}\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)
    
    # Create the retrieval chain using LCEL (LangChain Expression Language)
    retrieval_chain = (
        RunnablePassthrough.assign(
            retrieved_docs=lambda x: retriever.get_relevant_documents(x["question"])
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(x["retrieved_docs"]),
            document_sources=lambda x: list(set([
                doc.metadata.get('source_file', 'Unknown') 
                for doc in x["retrieved_docs"]
            ]))
        )
    )
    
    return retrieval_chain

def create_answer_chain():
    """Create an answer generation chain"""
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)
    
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based ONLY on the provided context.
        
        Rules:
        1. Answer ONLY from the context provided
        2. If you cannot find the answer in the context, say "I don't have enough information to answer this question"
        3. Always cite which source document(s) you're referencing
        4. Be concise but complete
        5. If multiple sources contain relevant information, synthesize them
        """),
        ("human", """Context from documents:
        {context}
        
        Question: {question}
        
        Answer based only on the context above:""")
    ])
    
    answer_chain = answer_prompt | llm | StrOutputParser()
    return answer_chain

def create_critic_chain():
    """Create a chain to refine and improve answers"""
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
    
    critic_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a critic that improves answers for clarity, accuracy, and completeness.
        
        Your job:
        1. Make the answer more clear and well-structured
        2. Ensure accuracy based on the provided context
        3. Improve readability and flow
        4. Do NOT add information not present in the context
        5. Keep source citations intact
        """),
        ("human", """Original Context:
        {context}
        
        Draft Answer to improve:
        {draft_answer}
        
        Provide an improved version of the answer:""")
    ])
    
    critic_chain = critic_prompt | llm | StrOutputParser()
    return critic_chain

# ================================
# Enhanced Agent Functions
# ================================
def make_groq(temp: float = 0.0):
    return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=temp)

def enhanced_retriever_agent(vectorstore: FAISS):
    """Enhanced retriever that uses chains"""
    retrieval_chain = create_retrieval_chain(vectorstore)
    
    def _node(state: AppState) -> AppState:
        # Add processing step
        steps = state.get("processing_steps", [])
        steps.append("🔍 Retrieving relevant documents...")
        state["processing_steps"] = steps
        
        # Use the retrieval chain
        result = retrieval_chain.invoke({"question": state["question"]})
        
        # Update state
        state["retrieved_docs"] = [
            {"text": d.page_content, "meta": d.metadata} 
            for d in result["retrieved_docs"]
        ]
        state["context"] = result["context"]
        state["document_sources"] = result["document_sources"]
        
        return state
    
    return _node

def enhanced_answer_agent():
    """Enhanced answer agent using chains"""
    answer_chain = create_answer_chain()
    
    def _node(state: AppState) -> AppState:
        steps = state.get("processing_steps", [])
        steps.append("💭 Generating initial answer...")
        state["processing_steps"] = steps
        
        # Generate answer using the chain
        draft_answer = answer_chain.invoke({
            "context": state.get("context", ""),
            "question": state["question"]
        })
        
        state["draft_answer"] = draft_answer
        return state
    
    return _node

def enhanced_critic_agent():
    """Enhanced critic agent using chains"""
    critic_chain = create_critic_chain()
    
    def _node(state: AppState) -> AppState:
        steps = state.get("processing_steps", [])
        steps.append("✨ Refining and improving answer...")
        state["processing_steps"] = steps
        
        # Improve answer using the chain
        final_answer = critic_chain.invoke({
            "context": state.get("context", ""),
            "draft_answer": state.get("draft_answer", "")
        })
        
        state["final_answer"] = final_answer
        
        # Calculate simple confidence score based on context relevance
        confidence = min(100, max(50, len(state.get("context", "")) / 100))
        state["confidence_score"] = confidence
        
        steps.append("✅ Processing complete!")
        state["processing_steps"] = steps
        
        return state
    
    return _node

# Build Enhanced Graph
def build_enhanced_graph(vectorstore: FAISS):
    """Build the enhanced multi-agent graph with chains"""
    graph = StateGraph(AppState)
    
    # Add nodes
    graph.add_node("retrieve", enhanced_retriever_agent(vectorstore))
    graph.add_node("answer", enhanced_answer_agent())
    graph.add_node("critic", enhanced_critic_agent())
    
    # Add edges (same linear flow)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "critic")
    graph.add_edge("critic", END)
    
    return graph.compile()

# ================================
# Enhanced Streamlit UI
# ================================
st.set_page_config(
    page_title="📚 Enhanced Multi-Document RAG",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Enhanced Multi-Document RAG System")
st.write("Upload multiple documents (PDF, TXT, CSV, DOCX) and ask questions using advanced chaining and retrieval!")

# Sidebar for document management
with st.sidebar:
    st.header("📁 Document Management")
    uploaded_files = st.file_uploader(
        "Upload your documents", 
        type=["pdf", "txt", "csv", "docx", "doc"],
        accept_multiple_files=True,
        help="You can upload multiple files of different types"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size} bytes)")

# Main content area
if uploaded_files:
    with st.spinner("🔄 Processing documents..."):
        # Process multiple documents
        doc_processor = DocumentProcessor()
        all_chunks, processed_files = doc_processor.process_multiple_files(uploaded_files)
        
        if not all_chunks:
            st.error("❌ No documents could be processed successfully")
            st.stop()
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        
        # Build enhanced graph with chains
        app_graph = build_enhanced_graph(vectorstore)
    
    st.success(f"✅ Successfully processed {len(processed_files)} documents with {len(all_chunks)} chunks")
    
    # Display processed documents
    with st.expander("📋 Processed Documents Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Processed Files:**")
            for i, file in enumerate(processed_files, 1):
                st.write(f"{i}. {file}")
        
        with col2:
            st.write("**Statistics:**")
            st.write(f"Total chunks: {len(all_chunks)}")
            st.write(f"Average chunk size: {sum(len(chunk.page_content) for chunk in all_chunks) // len(all_chunks)} characters")
    
    # Question input section
    st.header("💬 Ask Questions")
    
    # Sample questions based on document types
    sample_questions = [
        "What are the main topics covered in these documents?",
        "Can you summarize the key points?",
        "What specific information is available about...?",
        "Compare information from different documents"
    ]
    
    with st.expander("💡 Sample Questions"):
        for q in sample_questions:
            if st.button(q, key=f"sample_{q[:20]}"):
                st.session_state.question = q
    
    # Question input
    user_question = st.text_input(
        "Ask your question:", 
        value=st.session_state.get('question', ''),
        placeholder="e.g., What are the main findings in the research papers?"
    )
    
    if user_question:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            with st.spinner("🤖 Multi-agent processing in progress..."):
                # Initialize state
                state_in: AppState = {
                    "question": user_question,
                    "processing_steps": []
                }
                
                # Process through the enhanced graph
                state_out = app_graph.invoke(state_in)
            
            # Display results
            st.subheader("📝 Answer:")
            st.write(state_out.get("final_answer", "No answer generated"))
            
            # Additional information
            with st.expander("📊 Processing Details"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write("**Processing Steps:**")
                    for step in state_out.get("processing_steps", []):
                        st.write(step)
                
                with col_b:
                    st.write("**Sources Used:**")
                    for source in state_out.get("document_sources", []):
                        st.write(f"• {source}")
                    
                    confidence = state_out.get("confidence_score", 0)
                    st.metric("Confidence Score", f"{confidence:.1f}%")
            
            # Show retrieved context
            with st.expander("🔍 Retrieved Context"):
                retrieved_docs = state_out.get("retrieved_docs", [])
                for i, doc in enumerate(retrieved_docs, 1):
                    st.write(f"**Chunk {i}** (Source: {doc['meta'].get('source_file', 'Unknown')})")
                    st.write(doc["text"])
                    st.write("---")
        
        with col2:
            # Quick stats
            st.subheader("📈 Quick Stats")
            st.metric("Documents Searched", len(processed_files))
            st.metric("Chunks Retrieved", len(state_out.get("retrieved_docs", [])))
            st.metric("Context Length", len(state_out.get("context", "")))

else:
    st.info("👆 Please upload one or more documents to get started!")
    
    # Show supported formats
    st.subheader("📄 Supported Document Formats")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.write("**PDF Files**")
        st.write("• Research papers")
        st.write("• Reports")
        st.write("• Books")
    
    with col2:
        st.write("**Text Files**")
        st.write("• Notes")
        st.write("• Articles") 
        st.write("• Documentation")
    
    with col3:
        st.write("**CSV Files**")
        st.write("• Data tables")
        st.write("• Spreadsheets")
        st.write("• Structured data")
    
    with col4:
        st.write("**Word Documents**")
        st.write("• DOCX files")
        st.write("• DOC files")
        st.write("• Formatted text")

# Footer
st.markdown("---")
st.markdown("🚀 **Enhanced Features:** Multi-document support • Advanced chaining • Source tracking • Confidence scoring")