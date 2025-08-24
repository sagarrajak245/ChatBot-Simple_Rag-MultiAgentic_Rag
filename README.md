# RAG Systems Repository

A comprehensive collection of Retrieval Augmented Generation (RAG) implementations showcasing different approaches to document-based question answering.

## Quick Navigation

📂 **Individual System Documentation:**
- [Basic RAG Chatbot](https://github.com/sagarrajak245/ChatBot-Simple_Rag-MultiAgentic_Rag/blob/main/simpleRag.md) - Core RAG concepts and implementation
- [Multi-Agent RAG System](https://github.com/sagarrajak245/ChatBot-Simple_Rag-MultiAgentic_Rag/blob/main/MultiAgent-RagSystem.md) - Agent collaboration and orchestration  
- [Enhanced Multi-Document RAG](https://github.com/sagarrajak245/ChatBot-Simple_Rag-MultiAgentic_Rag/blob/main/Multi-Document_Rag.md) - Production-ready multi-format system

📋 **System Comparison Table** ⬇️

## What is RAG?

**RAG (Retrieval-Augmented Generation)** combines document retrieval with AI generation to answer questions based on your specific documents. Instead of relying solely on pre-trained knowledge, RAG systems:

1. Search through your documents for relevant information
2. Use that context to generate accurate, grounded answers
3. Provide source citations and transparency

## System Comparison

| Feature | Basic RAG | Multi-Agent RAG | Enhanced Multi-Document RAG |
|---------|-----------|-----------------|----------------------------|
| **Architecture** | Single LLM | 3 specialized agents | Multi-agent + LCEL chains |
| **Document Support** | PDF only | PDF only | PDF, TXT, CSV, DOCX |
| **Processing** | Direct Q&A | Research → Answer → Critique | Retrieval → Answer → Refinement |
| **UI Complexity** | Simple | Intermediate | Advanced |
| **Best For** | Learning RAG basics | Understanding agent collaboration | Production applications |

## 1. Basic RAG Chatbot

📚 **[View Complete Documentation →]([./basic-rag/README.md](https://github.com/sagarrajak245/ChatBot-Simple_Rag-MultiAgentic_Rag/blob/main/simpleRag.md))**

### Overview
A straightforward RAG implementation perfect for understanding core concepts.

### Key Features
- PDF document processing
- FAISS vector storage
- Groq LLaMA integration
- Streamlit web interface

### Architecture
```
PDF → Text Chunks → Embeddings → Vector Store → Question → Context Retrieval → LLM → Answer
```

### Use Cases
- Learning RAG fundamentals
- Simple document Q&A
- Proof of concept projects

### Quick Start
```bash
pip install streamlit langchain-community langchain-groq langchain-huggingface faiss-cpu
streamlit run basic_rag.py
```

## 2. Multi-Agent RAG System

📚 **[View Complete Documentation →]([./multi-agent-rag/README.md](https://github.com/sagarrajak245/ChatBot-Simple_Rag-MultiAgentic_Rag/blob/main/MultiAgent-RagSystem.md))**

### Overview
Demonstrates agent collaboration with specialized roles for research, answering, and critique.

### Key Features
- Three specialized AI agents
- LangGraph orchestration
- Process transparency
- Answer evolution tracking

### Architecture
```
Question → Research Agent → Answer Agent → Critic Agent → Final Answer
           (Find Context)   (Draft Answer)  (Refine & Polish)
```

### Agents
- **Research Agent**: Document retrieval and analysis specialist
- **Answer Agent**: Initial response generation focused on accuracy
- **Critic Agent**: Quality improvement and refinement expert

### Use Cases
- Understanding multi-agent systems
- Complex document analysis
- Quality-focused applications

### Quick Start
```bash
pip install streamlit langchain langgraph langchain-groq
streamlit run multi_agent_rag.py
```

## 3. Enhanced Multi-Document RAG

📚 **[View Complete Documentation →]([./enhanced-multi-doc-rag/README.md](https://github.com/sagarrajak245/ChatBot-Simple_Rag-MultiAgentic_Rag/blob/main/Multi-Document_Rag.md))**

### Overview
Production-ready system handling multiple document formats with advanced processing.

### Key Features
- Multi-format support (PDF, TXT, CSV, DOCX)
- LangChain Expression Language (LCEL)
- Advanced state management
- Confidence scoring
- Source tracking

### Architecture
```
Multiple Documents → Unified Processing → Vector Store → Multi-Agent Pipeline → Enhanced Results
```

### Processing Pipeline
1. **Document Processing**: Handle different file formats
2. **Intelligent Chunking**: Preserve context across splits
3. **Vector Storage**: Efficient similarity search
4. **Multi-Agent Processing**: Specialized retrieval and generation
5. **Result Enhancement**: Confidence scoring and source citation

### Use Cases
- Production document systems
- Multi-format document collections
- Enterprise applications

### Quick Start
```bash
pip install streamlit langchain langgraph langchain-groq langchain-huggingface faiss-cpu
streamlit run enhanced_rag.py
```

## Common Technologies

### Core Stack
- **LangChain**: Framework for LLM applications
- **Streamlit**: Web interface development
- **FAISS**: Vector similarity search
- **HuggingFace**: Text embeddings (sentence-transformers)
- **Groq**: High-performance LLM inference

### Document Processing
- **PyPDF2/PyPDFLoader**: PDF text extraction
- **TextLoader**: Plain text processing
- **CSVLoader**: Structured data handling
- **UnstructuredWordDocumentLoader**: Word document support

### AI Components
- **Embedding Model**: `all-MiniLM-L6-v2` for semantic search
- **LLM**: LLaMA-3.3-70B via Groq API
- **Vector Store**: FAISS for fast retrieval

## Installation & Setup

### Prerequisites
- Python 3.8+
- Groq API key (free at console.groq.com)
- 4GB+ RAM recommended

### Environment Setup
1. Clone the repository
2. Create `.env` file:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Universal Dependencies
```bash
pip install streamlit langchain langchain-community langchain-groq langchain-huggingface
pip install faiss-cpu python-dotenv pypdf unstructured
```

## Usage Patterns

### For Beginners
Start with **Basic RAG** to understand fundamental concepts:
1. Upload a PDF document
2. Ask simple questions
3. Observe retrieval and generation process

### For Developers
Progress to **Multi-Agent RAG** to learn about:
1. Agent specialization and collaboration
2. State management in complex workflows
3. Process transparency and debugging

### For Production
Use **Enhanced Multi-Document RAG** for:
1. Handling diverse document types
2. Building robust applications
3. Advanced features like confidence scoring

## Architecture Patterns

### Single-Agent Pattern (Basic RAG)
```
User Question → Document Search → Context Assembly → LLM Generation → Answer
```

### Multi-Agent Pattern (Agent RAG)
```
Question → Agent 1 (Research) → Agent 2 (Answer) → Agent 3 (Critique) → Final Answer
```

### Chain Pattern (Enhanced RAG)
```
Documents → Processing Chain → Retrieval Chain → Answer Chain → Critic Chain → Result
```

## Performance Considerations

### Document Processing
- **Chunk Size**: 500 characters with 50 overlap (optimal balance)
- **Retrieval Count**: 3-6 chunks per query
- **Similarity Threshold**: 0.1 (filters irrelevant content)

### Memory Usage
- Basic: ~2GB RAM
- Multi-Agent: ~3GB RAM  
- Enhanced: ~4GB RAM

### Response Time
- Basic: 2-5 seconds
- Multi-Agent: 5-10 seconds
- Enhanced: 3-8 seconds

## Customization Guide

### Embedding Models
```python
# Faster but lower quality
"sentence-transformers/all-MiniLM-L6-v2"

# Better quality but slower
"sentence-transformers/all-mpnet-base-v2"
```

### LLM Configuration
```python
# More creative responses
temperature=0.7

# More factual responses  
temperature=0.1
```

### Retrieval Tuning
```python
# More context (may include noise)
search_kwargs={"k": 8}

# Less context (more focused)
search_kwargs={"k": 3}
```

## Troubleshooting

### Common Issues
1. **API Key Error**: Ensure GROQ_API_KEY in .env file
2. **Memory Issues**: Reduce chunk size or document count
3. **Slow Processing**: Lower retrieval count or use smaller models
4. **Poor Answers**: Adjust similarity threshold or chunk overlap

### Performance Tips
- Use SSD storage for large document collections
- Monitor RAM usage during processing
- Consider GPU acceleration for embeddings
- Implement caching for repeated queries

## Learning Path

### Beginner (Basic RAG)
1. Understand RAG concepts
2. Learn document processing
3. Experiment with different questions
4. Modify chunk sizes and parameters

### Intermediate (Multi-Agent)
1. Study agent specialization
2. Understand state management
3. Learn orchestration patterns
4. Experiment with agent prompts

### Advanced (Enhanced System)
1. Master multi-format processing
2. Learn LCEL patterns
3. Implement custom chains
4. Build production features

## Contributing

Each system is designed to be:
- **Educational**: Clear code structure and documentation
- **Extensible**: Easy to modify and enhance
- **Production-Ready**: Error handling and optimization

### Areas for Contribution
- Additional document format support
- Alternative LLM integrations
- Enhanced UI components
- Performance optimizations
- Evaluation metrics

## License

MIT License - See individual system directories for specific details.

## Support

- Check system-specific documentation
- Review troubleshooting sections
- Join LangChain Discord community
- Create issues for bugs or feature requests

---

**Choose Your RAG Journey:**
- 🚀 **New to RAG?** → Start with Basic RAG
- 🤖 **Want Agent Collaboration?** → Try Multi-Agent RAG  
- 🏢 **Building Production Apps?** → Use Enhanced Multi-Document RAG

Each implementation builds upon the previous, creating a comprehensive learning and development path for RAG systems.
