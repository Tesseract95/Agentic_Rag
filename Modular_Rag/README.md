# Modular RAG Pipeline

A clean, modular implementation of Retrieval Augmented Generation with clear separation of concerns.

## 🏗️ Architecture

The pipeline is composed of independent, reusable modules:

```
Document → Preprocessor → Embeddings → Vector Store
                                            ↓
Query → Embeddings → Retriever → Response Generator → Answer
```

## 📦 Modules

### 1. `document_preprocessor.py`
**Purpose**: Text splitting and cleaning

**Features:**
- RecursiveCharacterTextSplitter integration
- Configurable chunk size and overlap
- Text cleaning and normalization
- File loading support

**Usage:**
```python
preprocessor = DocumentPreprocessor(chunk_size=200, chunk_overlap=50)
chunks = preprocessor.process_from_file("sample.txt")
```

### 2. `embeddings_module.py`
**Purpose**: Generate embeddings for text

**Features:**
- Google Gemini embeddings (gemini-embedding-001)
- 3072-dimensional vectors
- Batch processing support
- Error handling

**Usage:**
```python
embeddings = EmbeddingsModule()
vectors = embeddings.generate_embeddings_batch(chunks)
```

### 3. `vector_store.py`
**Purpose**: Pinecone vector database operations

**Features:**
- Index management
- Vector upsert operations
- Similarity search
- Index statistics

**Usage:**
```python
vector_store = VectorStore(index_name="my-index")
vector_store.upsert_vectors(chunks, embeddings)
```

### 4. `retriever.py`
**Purpose**: Semantic search and retrieval

**Features:**
- Query embedding generation
- Top-k similarity search
- Score-based ranking
- Context extraction

**Usage:**
```python
retriever = Retriever(vector_store, embeddings_module)
results = retriever.retrieve(query, top_k=3)
```

### 5. `response_generator.py`
**Purpose**: LLM-based response generation

**Features:**
- Gemini 2.5 Flash integration
- Customizable prompts
- Context-aware generation
- Score metadata

**Usage:**
```python
generator = ResponseGenerator()
response = generator.generate_response_with_scores(query, chunks)
```

### 6. `modular_rag_pipeline.py`
**Purpose**: Orchestrate all modules

**Features:**
- End-to-end pipeline
- Document ingestion
- Query processing
- Chat interface

## 🚀 Quick Start

### Installation

```bash
cd Modular_Rag
uv sync
```

### Basic Usage

```bash
# Run the pipeline
uv run python modular_rag_pipeline.py
```

### Custom Usage

```python
from modular_rag_pipeline import ModularRAGPipeline

# Initialize
rag = ModularRAGPipeline(index_name="my-custom-index")

# Ingest documents
rag.ingest_documents("my_document.txt")

# Query
result = rag.query("What is the main topic?", top_k=5)
print(result['answer'])

# Chat interface
rag.chat("Tell me more about X")
```

## 🔧 Configuration

### Pipeline Configuration

```python
rag = ModularRAGPipeline(
    index_name="my-index"  # Pinecone index name
)
```

### Module Configuration

```python
# Custom preprocessing
preprocessor = DocumentPreprocessor(
    chunk_size=500,      # Larger chunks
    chunk_overlap=100    # More overlap
)

# Custom response generation
generator = ResponseGenerator(
    model_name="gemini-2.5-flash",
    temperature=0.3      # More deterministic
)
```

## 📊 Output Format

### Query Response

```python
{
    'answer': 'The answer to your question...',
    'query': 'Your original question',
    'sources': ['chunk1', 'chunk2', 'chunk3'],
    'scores': [0.95, 0.87, 0.82],
    'response': 'Generated response text',
    'num_sources': 3,
    'max_score': 0.95,
    'min_score': 0.82,
    'avg_score': 0.88
}
```

## 🧪 Testing

### Test All Modules

```bash
uv run python -c "
import document_preprocessor
import embeddings_module
import vector_store
import retriever
import response_generator
print('✅ All modules imported successfully')
"
```

### Test Individual Modules

```bash
# Test preprocessing
uv run python -c "
from document_preprocessor import DocumentPreprocessor
dp = DocumentPreprocessor()
chunks = dp.process_from_file('sample.txt')
print(f'Created {len(chunks)} chunks')
"
```

## 📁 Files

- `document_preprocessor.py` - Text processing
- `embeddings_module.py` - Embedding generation
- `vector_store.py` - Pinecone operations
- `retriever.py` - Semantic search
- `response_generator.py` - LLM generation
- `modular_rag_pipeline.py` - Pipeline orchestrator
- `modular_rag_pipeline_with_chromaDb.py` - ChromaDB variant
- `vector_store_chromadb.py` - ChromaDB implementation
- `test_modular_rag.py` - Test script
- `sample.txt` - Sample document
- `pyproject.toml` - Dependencies

## 🎯 Use Cases

- **Learning**: Understand RAG components individually
- **Production**: Use as foundation for production systems
- **Experimentation**: Swap modules (e.g., different vector stores)
- **Customization**: Easy to modify individual components

## 🔄 ChromaDB Variant

Also includes a ChromaDB implementation:

```bash
uv run python modular_rag_pipeline_with_chromaDb.py
```

**Benefits:**
- Local vector storage (no API keys)
- Faster for small datasets
- Simpler setup

## 🛠️ Extending

### Add a New Vector Store

1. Create `vector_store_newdb.py` implementing same interface
2. Update pipeline to use new store
3. No changes needed to other modules!

### Add Custom Embeddings

1. Modify `embeddings_module.py`
2. Update dimension in vector store
3. Pipeline works the same!

## 📝 Environment Variables

```bash
GOOGLE_API_KEY=your_google_api_key
PC_API_KEY=your_pinecone_api_key
```

## 🐛 Troubleshooting

**Issue**: "No module named 'langchain_text_splitters'"
- **Fix**: Run `uv sync` to install dependencies

**Issue**: "Pinecone index not found"
- **Fix**: Create index with dimension=3072, metric=cosine

**Issue**: "No API key found"
- **Fix**: Create `.env` file in project root with API keys

## 📚 Dependencies

- `python-dotenv` - Environment variables
- `langchain` - LangChain framework
- `langchain-google-genai` - Gemini integration
- `langchain-pinecone` - Pinecone integration
- `langchain-text-splitters` - Text splitting
- `langchain-core` - Core components
- `pinecone` - Pinecone client
- `chromadb` - ChromaDB (optional)

## 🎓 Learning Resources

- Compare with `Naive_Rag.py` in parent directory to see benefits
- Review `Single_agentic-rag/` for extended documentation
- Check `modular_rag_architecture.md` for architecture deep-dive

---

**Clean Architecture • Production Ready • Easy to Extend**
