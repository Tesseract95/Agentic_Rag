# Quick Start Guide

Get started with any RAG implementation in under 5 minutes!

## 🏃 Prerequisites

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Get API Keys**:
   - **Google Gemini**: [Get API Key](https://makersuite.google.com/app/apikey)
   - **Pinecone**: [Sign up](https://www.pinecone.io/) and get API key
   - **Tavily** (optional, for CRAG): [Get API Key](https://tavily.com/)

## ⚡ 5-Minute Setup

### 1. Clone/Navigate to Project

```bash
cd "/path/to/code files"
```

### 2. Create Environment File

```bash
cp .env.example .env
```

Edit `.env`:
```bash
GOOGLE_API_KEY=your_actual_google_api_key
PC_API_KEY=your_actual_pinecone_api_key
TAVILY_API_KEY=your_actual_tavily_api_key  # Optional
```

### 3. Choose Your RAG Implementation

## 🎯 Option 1: Naive RAG (Simplest)

**Best for**: Learning basics, quick prototyping

```bash
# Install dependencies
uv sync

# Run
uv run python Naive_Rag.py
```

**Expected Output**:
```
✅ Connected to existing index: nikhil-agent-module-gemini
📄 Processing sample.txt
✅ Stored 15 document chunks

🔍 Query: What is the transformer model?
💡 Answer: The Transformer is a neural network...
📊 Relevance Scores: ['0.892', '0.845', '0.821']
```

---

## 🎯 Option 2: Modular RAG (Recommended for Production)

**Best for**: Production apps, clean architecture

```bash
# Navigate and install
cd Modular_Rag
uv sync

# Run
uv run python modular_rag_pipeline.py
```

**Features**:
- Clean module separation
- Easy to customize
- Production-ready

---

## 🎯 Option 3: CRAG (Most Intelligent)

**Best for**: High-accuracy needs, dynamic information

```bash
# Install dependencies
cd ..
uv sync

# Run
uv run python CRAG.py
```

**Features**:
- Self-correcting
- Web search fallback
- Quality assessment

---

## 🎯 Option 4: Multi-Agentic RAG (Most Advanced)

**Best for**: Multi-domain systems, e-commerce

```bash
# Navigate and install
cd Mullti_Agentic_Rag
uv sync

# First run: Ingest data
INGEST=true uv run python multi_agent_rag.py

# Subsequent runs: Query
uv run python multi_agent_rag.py
```

**Features**:
- 5 specialized agents
- Domain-specific routing
- Multi-index retrieval

---

## 🔍 Test Without API Keys

Want to test imports without setting up API keys?

```bash
cd Modular_Rag
uv run python -c "
import document_preprocessor
import embeddings_module
import vector_store
import retriever
import response_generator
print('✅ All modules work!')
"
```

---

## 🧪 Custom Query

All implementations support custom queries:

### Naive RAG
```python
from Naive_Rag import NaiveRAG, rag

result = rag.query("Your custom question?")
print(result['answer'])
```

### Modular RAG
```python
from modular_rag_pipeline import ModularRAGPipeline

rag = ModularRAGPipeline()
result = rag.chat("Your question?")
```

### CRAG
```python
from CRAG import CRAGSystem

crag = CRAGSystem()
result = crag.query("Your question?")
print(result['final_answer'])
```

### Multi-Agentic RAG
```python
from multi_agent_rag import MultiAgentRAG

rag = MultiAgentRAG()
result = rag.invoke("Your question?")
print(result['answer'])
```

---

## 🐛 Common Issues

### Issue: "PC_API_KEY not found"

**Solution**:
```bash
# Check .env file exists
ls -la .env

# Verify content
cat .env
```

### Issue: "Index not found"

**Solutions**:

**For Naive/CRAG/Modular RAG:**
1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Create index:
   - Name: `nikhil-agent-module-gemini`
   - Dimension: `3072`
   - Metric: `cosine`

**For Multi-Agentic RAG:**
Run ingestion first:
```bash
INGEST=true uv run python multi_agent_rag.py
```

### Issue: "Module not found"

**Solution**:
```bash
# Make sure you're in the right directory
pwd

# Sync dependencies
uv sync
```

### Issue: "uv: command not found"

**Solution**:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart terminal
```

---

## 📊 Compare All Implementations

| Feature | Naive RAG | Modular RAG | CRAG | Multi-Agentic |
|---------|-----------|-------------|------|---------------|
| Setup Time | 2 min | 3 min | 4 min | 5 min |
| Complexity | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Accuracy | Good | Good | Excellent | Excellent |
| Web Search | ❌ | ❌ | ✅ | ❌ |
| Multi-Domain | ❌ | ❌ | ❌ | ✅ |
| Production Ready | ❌ | ✅ | ✅ | ✅ |

---

## 🎓 Learning Path

**Day 1**: Start with **Naive RAG**
- Understand basic RAG flow
- Learn vector embeddings
- Practice with Pinecone

**Day 2**: Move to **Modular RAG**
- Learn clean architecture
- Understand module separation
- Explore customization

**Day 3**: Try **CRAG**
- Learn self-correction
- Understand LangGraph
- Add web search

**Day 4**: Master **Multi-Agentic RAG**
- Multi-agent orchestration
- Domain routing
- Production patterns

---

## 🚀 Next Steps

1. **Read the main README**: `README.md`
2. **Explore module docs**: Each folder has its own README
3. **Customize**: Modify chunk size, models, prompts
4. **Build**: Create your own RAG application!

---

## 💡 Tips

- Start simple, add complexity gradually
- Use Modular RAG as foundation for custom apps
- CRAG when accuracy is critical
- Multi-Agentic for multi-domain systems

---

## 📞 Need Help?

Check individual README files:
- `Modular_Rag/README.md`
- `Mullti_Agentic_Rag/README.md`
- `Single_agentic-rag/README.md`

Or review architecture docs in project folders.

---

**Happy RAG Building! 🎉**
