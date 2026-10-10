"""
Helper script to create Pinecone index with correct dimensions for HuggingFace embeddings
"""
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

def create_index(index_name="swastik-agent-module-gemini", dimension=384):
    """
    Create a Pinecone index for HuggingFace embeddings
    
    Args:
        index_name: Name of the index to create
        dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Check if index already exists
    existing_indexes = pc.list_indexes().names()
    
    if index_name in existing_indexes:
        print(f"⚠️  Index '{index_name}' already exists!")
        print(f"📊 Current index stats:")
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"   - Dimension: {stats.dimension if hasattr(stats, 'dimension') else 'N/A'}")
        print(f"   - Total vectors: {stats.total_vector_count}")
        
        response = input(f"\n🗑️  Do you want to delete and recreate it? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            print(f"🗑️  Deleting existing index...")
            pc.delete_index(index_name)
            print(f"✅ Deleted!")
        else:
            print("❌ Keeping existing index. Exiting.")
            return
    
    # Create new index
    print(f"\n🛠️  Creating Pinecone index: {index_name}")
    print(f"   - Dimension: {dimension} (all-MiniLM-L6-v2)")
    print(f"   - Metric: cosine")
    print(f"   - Cloud: AWS")
    print(f"   - Region: us-east-1")
    
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    
    print(f"✅ Index '{index_name}' created successfully!")
    print(f"\n📝 You can now run: uv run python test_modular_rag.py")

if __name__ == "__main__":
    print("="*60)
    print("🔧 Pinecone Index Creator")
    print("="*60)
    
    # Get index name from user or use default
    index_name = input("\nEnter index name (default: swastik-agent-module-gemini): ").strip()
    if not index_name:
        index_name = "swastik-agent-module-gemini"
    
    create_index(index_name)
