# 📄 RAG Chatbot with Groq & FAISS

## 🎯 What is RAG?

**RAG (Retrieval-Augmented Generation)** is a powerful AI technique that combines the best of two worlds:
- **Retrieval**: Finding relevant information from your documents
- **Generation**: Using AI to create human-like responses based on that information

Think of RAG as having a super-smart research assistant that can:
1. Instantly search through your documents
2. Find the most relevant information
3. Generate a natural, conversational answer based on what it found

## 🏗️ How This RAG System Works

```
📄 PDF Document
       ↓
   [Document Loader]
       ↓
📝 Raw Text Content
       ↓
   [Text Splitter]
       ↓
🧩 Text Chunks
       ↓
   [Embeddings Model]
       ↓
🔢 Vector Representations
       ↓
   [FAISS Vector Store]
       ↓
💾 Searchable Knowledge Base

When you ask a question:
❓ User Question → [Retriever] → 📋 Relevant Chunks → [LLM + Prompt] → ✨ Final Answer
```

## 🧠 Key Components Explained

### 1. Document Loading & Processing
```python
# Load PDF and split into manageable chunks
loader = PyPDFLoader(temp_pdf_path)
docs = loader.load()
```
**Why chunking?** Large documents are split into smaller pieces because:
- AI models have input length limits
- Smaller chunks provide more focused, relevant context
- Better retrieval accuracy

### 2. Embeddings (The Magic Behind Search)
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```
**What are embeddings?** Think of embeddings as "DNA fingerprints" for text:
- Convert text into numerical vectors (lists of numbers)
- Similar text gets similar numbers
- Enables semantic search (finding meaning, not just keywords)

**Example:**
- "The cat is sleeping" → [0.2, 0.8, 0.1, ...]
- "A feline is resting" → [0.3, 0.7, 0.2, ...] (similar numbers!)
- "I love pizza" → [0.9, 0.1, 0.8, ...] (very different numbers)

### 3. Vector Store (Your Smart Database)
```python
vectorstore = FAISS.from_documents(chunks, embeddings)
```
**FAISS** is like a super-efficient librarian that:
- Stores all your document chunks as vectors
- Quickly finds the most similar chunks to your question
- Uses advanced algorithms for lightning-fast search

### 4. The Retrieval Process
```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Get top 3 most relevant chunks
)
```

### 5. Language Model (The Answer Generator)
```python
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0  # Low temperature = more focused, factual responses
)
```

## 🔄 The Complete RAG Flow

### Step-by-Step Process:

1. **Document Ingestion**
   ```
   PDF → Text Extraction → Split into Chunks
   ```

2. **Vectorization**
   ```
   Text Chunks → Embedding Model → Vector Representations → Store in FAISS
   ```

3. **Query Processing**
   ```
   User Question → Convert to Vector → Search Similar Vectors in FAISS
   ```

4. **Context Retrieval**
   ```
   Find Top-K Similar Chunks → Extract Original Text → Combine as Context
   ```

5. **Answer Generation**
   ```
   Context + Question + Prompt → Language Model → Final Answer
   ```

## 📋 Installation & Setup

### Prerequisites
```bash
pip install streamlit langchain-community langchain-groq langchain-huggingface
pip install faiss-cpu PyPDF2 python-dotenv sentence-transformers
```

### Environment Setup
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

### Running the Application
```bash
streamlit run app.py
```

## 🎛️ Configuration Parameters

### Text Splitting Parameters
```python
chunk_size=500      # Characters per chunk (balance: too small = loss of context, too large = noise)
chunk_overlap=50    # Overlap between chunks (preserves context across boundaries)
```

### Retrieval Parameters
```python
search_kwargs={"k": 3}  # Number of chunks to retrieve (more = more context but potential noise)
```

### LLM Parameters
```python
temperature=0  # 0 = deterministic, 1 = creative (for factual QA, keep low)
```

## 🎨 Architecture Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   📄 PDF File   │────│  Document Loader │────│  📝 Raw Text    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                       ┌──────────────────┐    ┌─────────────────┐
                       │  🧩 Text Chunks  │────│  Text Splitter  │
                       └──────────────────┘    └─────────────────┘
                                │
                       ┌──────────────────┐    
                       │ 🤖 Embeddings    │    
                       │     Model        │    
                       └──────────────────┘    
                                │
                       ┌──────────────────┐    
                       │ 💾 FAISS Vector  │    
                       │     Store        │    
                       └──────────────────┘    
                                │
    ❓ User Question ──────────────────────────────┐
                                                  │
    ┌─────────────────┐    ┌──────────────────┐   │   ┌─────────────────┐
    │ ✨ Final Answer │────│ 🧠 Language Model │───┼───│ 🔍 Retriever    │
    └─────────────────┘    └──────────────────┘   │   └─────────────────┘
                                   │               │            │
                           ┌──────────────────┐    │   ┌─────────────────┐
                           │ 📋 Custom Prompt │────┘   │ 📄 Top-K Chunks │
                           └──────────────────┘        └─────────────────┘
```

## 🚀 Usage Examples

### Example 1: Technical Document
**Question:** "What is the maximum operating temperature?"
**Process:** 
- System searches for chunks containing temperature specifications
- Finds: "The device operates between -10°C to 85°C..."
- **Answer:** "The maximum operating temperature is 85°C."

### Example 2: Policy Document
**Question:** "What is the refund policy?"
**Process:**
- Retrieves chunks about refunds and returns
- Combines relevant information
- **Answer:** "According to the document, refunds are processed within 30 days..."

## 🔧 Customization Options

### 1. Different Embedding Models
```python
# For better quality (slower)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# For faster processing (lower quality)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### 2. Custom Prompts
```python
prompt_template = """
You are an expert in [DOMAIN]. Answer based on the context provided.
Be specific and cite relevant sections when possible.

Context: {context}
Question: {question}
Answer:
"""
```

### 3. Different Vector Stores
```python
# Alternative to FAISS
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(chunks, embeddings)
```

## 📊 Performance Tips

### Optimizing Chunk Size
- **Small chunks (100-300 chars)**: More precise but may lose context
- **Medium chunks (300-800 chars)**: Good balance for most use cases
- **Large chunks (800+ chars)**: More context but may include irrelevant info

### Optimizing Retrieval
- **k=1-3**: For focused, specific answers
- **k=4-7**: When questions might need broader context
- **k=8+**: For comprehensive analysis (may introduce noise)

## 🐛 Common Issues & Solutions

### Issue 1: "I don't know" responses
**Cause:** Information not found in retrieved chunks
**Solution:** 
- Reduce chunk size for more granular search
- Increase k value to retrieve more chunks
- Check if information actually exists in the document

### Issue 2: Slow performance
**Cause:** Large documents or too many chunks
**Solution:**
- Increase chunk size to reduce total chunks
- Use a lighter embedding model
- Consider document preprocessing

### Issue 3: Inaccurate answers
**Cause:** Retrieved chunks contain wrong context
**Solution:**
- Improve chunk splitting strategy
- Fine-tune retrieval parameters
- Enhance prompt engineering

## 🎓 Learning Path

### Beginner Level
1. Understand what RAG is and why it's useful
2. Run the basic application
3. Try different types of questions
4. Experiment with different PDFs

### Intermediate Level
1. Modify chunk sizes and observe effects
2. Try different embedding models
3. Customize the prompt template
4. Add conversation memory

### Advanced Level
1. Implement hybrid search (keyword + semantic)
2. Add metadata filtering
3. Build multi-document RAG
4. Implement evaluation metrics

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [Groq API Documentation](https://console.groq.com/docs)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy RAG Building! 🚀**