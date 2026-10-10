from pinecone import Pinecone
from dotenv import load_dotenv
import os
from typing import List, Dict
import uuid
import json

# LangChain imports
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to existing index
index_name = "nikhil-agent-module-gemini"
index = pc.Index(index_name)

print(f"✅ Connected to existing index: {index_name}")

# Initialize Google Gemini
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.7
)

# Initialize Gemini Embeddings
# Using gemini-embedding-001 which produces 3072 dimensions
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

print("✅ Successfully connected to Pinecone!")
print(f"Index stats: {index.describe_index_stats()}")

class NaiveRAG:
    def __init__(self, pinecone_index, embeddings_model, llm_model):
        self.index = pinecone_index
        self.embeddings = embeddings_model
        self.llm = llm_model
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            length_function=len,
        )
        
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks using LangChain"""
        try:
            documents = self.text_splitter.create_documents([text])
            chunks = [doc.page_content for doc in documents]
            return chunks
        except Exception as e:
            print(f"Error chunking text: {e}")
            return []
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Gemini"""
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def store_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Store document chunks in Pinecone"""
        if metadata is None:
            metadata = [{}] * len(documents)
        
        vectors_to_upsert = []
        
        for i, doc in enumerate(documents):
            embedding = self.get_embedding(doc)
            if embedding:
                vector_id = str(uuid.uuid4())
                vectors_to_upsert.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': {
                        'text': doc,
                        'chunk_index': i,
                        **metadata[i]
                    }
                })
        
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)
            print(f"✅ Stored {len(vectors_to_upsert)} document chunks")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve most relevant chunks for a query"""
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            return [
                {
                    'text': match.metadata['text'],
                    'score': match.score,
                    'metadata': match.metadata
                }
                for match in results.matches
            ]
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []
    
    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """Generate answer using retrieved context with Gemini"""
        context = "\n\n".join(context_chunks)
        
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Answer the question using ONLY the information from the context below. 
DO NOT use any prior knowledge. If the context doesn't contain the answer, 
say "I cannot answer based on the provided information."

Context:
{context}

Question: {question}

Answer:"""
        )
        
        try:
            prompt = prompt_template.format(context=context, question=query)
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """Main RAG query function"""
        print(f"\n🔍 Query: {question}")
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(question, top_k)
        
        if not relevant_chunks:
            return {
                'answer': 'No relevant information found.',
                'sources': [],
                'scores': []
            }
        
        # Extract text from chunks
        context_texts = [chunk['text'] for chunk in relevant_chunks]
        scores = [chunk['score'] for chunk in relevant_chunks]
        
        # Generate answer
        answer = self.generate_answer(question, context_texts)
        
        return {
            'answer': answer,
            'sources': context_texts,
            'scores': scores
        }

# Initialize RAG system
rag = NaiveRAG(index, embeddings, llm)

# Load and process the sample text
def load_and_process_documents():
    """Load sample.txt and process it for RAG"""
    try:
        with open('sample.txt', 'r') as file:
            text = file.read()
        
        print("📄 Processing sample.txt...")
        chunks = rag.chunk_text(text)
        print(f"📝 Created {len(chunks)} chunks using LangChain text splitter")
        
        # Store in Pinecone
        rag.store_documents(chunks)
        
        return True
    except FileNotFoundError:
        print("❌ sample.txt not found")
        return False
    except Exception as e:
        print(f"❌ Error processing documents: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Process documents
    if load_and_process_documents():
        print("\n" + "="*50)
        print("🤖 Naive RAG System Ready!")
        print("📄 Processing sample.txt")
        print("="*50)
        
        # Example queries
        example_queries = [
            "What is the transformer model?"
        ]
        
        for query in example_queries:
            result = rag.query(query)
            print(f"\n❓ Question: {query}")
            print(f"💡 Answer: {result['answer']}")
            print(f"📊 Relevance Scores: {[f'{score:.3f}' for score in result['scores']]}")
            print("-" * 50)