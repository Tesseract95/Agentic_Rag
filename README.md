# Agentic RAG Systems Collection

A comprehensive collection of Retrieval Augmented Generation (RAG) implementations ranging from naive approaches to advanced multi-agent systems with corrective mechanisms.

## 📁 Project Structure

```
agentic_rag/
├── Naive_Rag.py                    # Basic RAG implementation
├── CRAG.py                         # Corrective RAG with quality assessment
├── CRAG_with_memory_checkpointer.py # CRAG with session memory
├── Simple_Rag_Pipeline_using_langgraph.py # LangGraph RAG pipeline
├── Modular_Rag/                    # Modular RAG with clean architecture
├── Single_agentic-rag/             # Single-agent RAG system
└── Mullti_Agentic_Rag/             # Multi-agent RAG across domains
```

## 🚀 Quick Start

### Prerequisites

1. **Install uv** (Python package manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys:
   # - GOOGLE_API_KEY (for Gemini LLM and embeddings)
   # - PC_API_KEY (for Pinecone vector database)
   # - TAVILY_API_KEY (for web search in CRAG)
   ```

### Installation

Each project can be run independently using `uv`:

```bash
# Root level (for Naive_Rag.py, CRAG.py, etc.)
cd "code files"
uv sync

# Modular RAG
cd Modular_Rag
uv sync

# Single Agentic RAG
cd Single_agentic-rag
uv sync

# Multi-Agentic RAG
cd Mullti_Agentic_Rag
uv sync
```

## 📚 RAG Implementations

### 1. Naive RAG (`Naive_Rag.py`)

Basic RAG implementation with:
- Text chunking using LangChain's RecursiveCharacterTextSplitter
- Gemini embeddings (gemini-embedding-001)
- Pinecone vector storage
- Simple retrieval and generation

**Run:**
```bash
uv run python Naive_Rag.py
```

**Use Cases:**
- Learning RAG fundamentals
- Quick prototyping
- Simple Q&A over documents

---

### 2. CRAG - Corrective RAG (`CRAG.py`)

Advanced RAG with self-correction mechanisms using LangGraph:

**Features:**
- **Quality Assessment Agent**: Evaluates retrieval relevance and factual consistency
- **Web Search Integration**: Uses Tavily to fetch additional context when needed
- **Iterative Refinement**: Regenerates answers if initial quality is poor
- **Multi-stage Pipeline**: Retrieve → Generate → Assess → Correct → Finalize

**Run:**
```bash
uv run python CRAG.py
```

**Architecture:**
```
Query → Retrieve → Generate Draft → Assess Quality
                                          ↓
                                   [If Poor Quality]
                                          ↓
                              Web Search → Regenerate
                                          ↓
                                   Final Answer
```

**Use Cases:**
- High-accuracy requirements
- Dynamic information needs
- Quality-critical applications

---

### 3. CRAG with Memory (`CRAG_with_memory_checkpointer.py`)

Enhanced CRAG with session management:

**Additional Features:**
- Session-based conversation tracking
- LangGraph MemorySaver checkpointer
- Multi-turn conversation support
- Session history retrieval

**Run:**
```bash
uv run python CRAG_with_memory_checkpointer.py
```

**Use Cases:**
- Conversational RAG applications
- Multi-turn Q&A systems
- Session-aware assistants

---

### 4. Simple RAG with LangGraph (`Simple_Rag_Pipeline_using_langgraph.py`)

Student data RAG demonstrating LangGraph orchestration:

**Features:**
- Graph-based pipeline
- Structured data handling (CSV)
- Clean node-based architecture

**Run:**
```bash
uv run python Simple_Rag_Pipeline_using_langgraph.py
```

---

### 5. Modular RAG (`Modular_Rag/`)

Clean, modular architecture with separation of concerns:

**Modules:**
- `document_preprocessor.py` - Text splitting and cleaning
- `embeddings_module.py` - Embedding generation
- `vector_store.py` - Pinecone vector operations
- `retriever.py` - Semantic search
- `response_generator.py` - LLM response generation
- `modular_rag_pipeline.py` - Pipeline orchestrator

**Run:**
```bash
cd Modular_Rag
uv run python modular_rag_pipeline.py
```

**Also includes ChromaDB variant:**
```bash
uv run python modular_rag_pipeline_with_chromaDb.py
```

**Architecture Benefits:**
- Easy to test individual components
- Swap implementations (Pinecone ↔ ChromaDB)
- Reusable modules
- Clear responsibilities

**Use Cases:**
- Production applications
- Custom RAG pipelines
- Component experimentation

---

### 6. Single Agentic RAG (`Single_agentic-rag/`)

Same modular architecture as Modular_Rag with additional documentation:

**Run:**
```bash
cd Single_agentic-rag
uv run python modular_rag_pipeline.py
```

Includes comprehensive architecture documentation in `modular_rag_architecture.md`.

---

### 7. Multi-Agentic RAG (`Mullti_Agentic_Rag/`)

Advanced multi-agent system with domain-specific knowledge bases:

**Agents:**
1. **Orchestrator**: Routes queries to appropriate domains (orders/products/support)
2. **Researcher**: Retrieves from domain-specific Pinecone indexes
3. **Analyst**: Extracts key facts from retrieved context
4. **Critic**: Assesses completeness and identifies gaps
5. **Synthesizer**: Produces final coherent answer

**Domains:**
- Orders (order.csv → orders-index)
- Products (product.csv → products-index)
- Support (support_catalogue.csv → support-index)

**Run:**

1. **Ingest data** (first time only):
   ```bash
   cd Mullti_Agentic_Rag
   INGEST=true uv run python multi_agent_rag.py
   ```

2. **Query the system**:
   ```bash
   uv run python multi_agent_rag.py
   ```

**Architecture:**
```
Query → Orchestrator → Researcher (multiple indexes)
                            ↓
                        Analyst → Critic → Synthesizer
                                            ↓
                                      Final Answer
```

**Use Cases:**
- Multi-domain knowledge systems
- E-commerce support systems
- Complex enterprise RAG
- Domain-specific routing

---

## 🔧 Configuration

### Environment Variables

```bash
# Required
GOOGLE_API_KEY=your_key_here          # Gemini LLM & embeddings
PC_API_KEY=your_key_here              # Pinecone vector database

# Optional
TAVILY_API_KEY=your_key_here          # Web search (CRAG only)
PC_CLOUD=aws                          # Pinecone cloud (default: aws)
PC_REGION=us-east-1                   # Pinecone region (default: us-east-1)
INGEST=false                          # Trigger data ingestion
```

### Pinecone Indexes

Different implementations use different indexes:

- **Naive RAG, CRAG**: `nikhil-agent-module-gemini` (dimension: 3072, Gemini embeddings)
- **Modular RAG, Single Agentic**: `nikhil-agent-module-gemini` (configurable)
- **Multi-Agentic RAG**:
  - `orders-index` (dimension: 384, sentence-transformers)
  - `products-index` (dimension: 384, sentence-transformers)
  - `support-index` (dimension: 384, sentence-transformers)

## 📊 Comparison Matrix

| Implementation | Complexity | Agents | Web Search | Memory | Best For |
|---------------|-----------|--------|-----------|--------|----------|
| Naive RAG | ⭐ | 1 | ❌ | ❌ | Learning, prototyping |
| CRAG | ⭐⭐⭐ | 1 | ✅ | ❌ | High accuracy needs |
| CRAG + Memory | ⭐⭐⭐⭐ | 1 | ✅ | ✅ | Conversational AI |
| Modular RAG | ⭐⭐ | 1 | ❌ | ❌ | Production apps |
| Single Agentic | ⭐⭐ | 1 | ❌ | ❌ | Clean architecture |
| Multi-Agentic | ⭐⭐⭐⭐⭐ | 5 | ❌ | ❌ | Multi-domain systems |

## 🧪 Testing

Each project has been tested with `uv run`:

```bash
# Test imports (Modular_Rag example)
cd Modular_Rag
uv run python -c "
import document_preprocessor
import embeddings_module
import vector_store
import retriever
import response_generator
print('✅ All modules imported successfully')
"
```

## 📖 Learning Path

**Recommended order:**

1. **Naive_Rag.py** - Understand RAG basics
2. **Modular_Rag/** - Learn clean architecture
3. **Simple_Rag_Pipeline_using_langgraph.py** - Understand LangGraph
4. **CRAG.py** - Learn corrective mechanisms
5. **CRAG_with_memory_checkpointer.py** - Add session management
6. **Mullti_Agentic_Rag/** - Master multi-agent systems

## 🛠️ Development

### Project Management with uv

```bash
# Add a new dependency
uv add package-name

# Run without installing
uv run --no-project python script.py

# Update dependencies
uv sync

# Create lock file
uv lock
```

### Code Structure Best Practices

The Modular RAG implementations demonstrate:
- **Single Responsibility**: Each module has one clear purpose
- **Dependency Injection**: Components are injected, not hardcoded
- **Configuration**: Settings exposed via constructors
- **Error Handling**: Graceful degradation with informative messages
- **Logging**: Print statements for observability

## 🐛 Troubleshooting

### Common Issues

1. **Module not found errors**:
   ```bash
   cd <project_folder>
   uv sync
   ```

2. **API key errors**:
   - Verify `.env` file exists and contains valid keys
   - Check key format (no quotes needed in .env)

3. **Pinecone index errors**:
   - Ensure index exists with correct dimensions
   - Check index name matches configuration

4. **Import errors** (langchain):
   - Use `langchain_text_splitters` not `langchain.text_splitter`
   - Use `langchain_core.prompts` not `langchain.prompts`

## 📝 License

This project is for educational purposes. See individual files for specific licensing information.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional RAG patterns
- More vector store integrations
- Enhanced evaluation metrics
- Performance benchmarks

## 🔗 References

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Google Gemini API](https://ai.google.dev/)
- [Corrective RAG Paper](https://arxiv.org/abs/2401.15884)

---

**Built with ❤️ for learning advanced RAG patterns**

For questions or issues, please refer to the individual README files in each subdirectory.
