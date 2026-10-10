import os
from typing import List, Dict, Any, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from pinecone import Pinecone
from tavily import TavilyClient
import json

from dotenv import load_dotenv

load_dotenv()

# Initialize clients
class CRAGState(BaseModel):
    query: str
    retrieved_docs: List[Document] = []
    draft_answer: str = ""
    assessment: Dict[str, Any] = {}
    corrective_action: str = ""
    final_answer: str = ""
    web_search_results: List[Dict] = []

# Initialize components
def initialize_components():
    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Pinecone
    pc = Pinecone(api_key=os.getenv("PC_API_KEY"))
    index_name = "nikhil-agent-module-gemini"
    
    # Vector Store
    vector_store = PineconeVectorStore(
        index=pc.Index(index_name),
        embedding=embeddings,
        text_key="text"
    )
    
    # Tavily for web search
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    return llm, vector_store, tavily_client

# Standard RAG Component
class StandardRAG:
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm
        
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve documents from vector store"""
        return self.vector_store.similarity_search(query, k=k)
    
    def generate(self, query: str, documents: List[Document]) -> str:
        """Generate draft answer from retrieved documents"""
        context = "\n\n".join([doc.page_content for doc in documents])
        
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful AI assistant. Based on the following context, answer the user's question.
        
        Context:
        {context}
        
        Question: {question}
        
        Provide a comprehensive and accurate answer based solely on the provided context.
        If the context doesn't contain enough information to fully answer the question, 
        acknowledge this in your response.
        
        Answer:
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": query})

# Corrective Agent using LangGraph
class CorrectiveAgent:
    def __init__(self, llm, tavily_client):
        self.llm = llm
        self.tavily_client = tavily_client
        
    def assess_quality(self, state: CRAGState) -> Dict[str, Any]:
        """Assess the quality of retrieval and draft answer"""
        
        assessment_prompt = ChatPromptTemplate.from_template("""
        You are a skeptical quality assessment agent. Your job is to critically evaluate the RAG system's performance.
        
        QUERY: {query}
        
        RETRIEVED DOCUMENTS:
        {documents}
        
        DRAFT ANSWER:
        {draft_answer}
        
        Assess the following aspects:
        
        1. RETRIEVAL RELEVANCE: Are these the best documents for the query? Rate as Good/Poor.
        2. FACTUAL CONSISTENCY: Does the generated answer correctly reflect the retrieved documents? Rate as Good/Poor.
        3. COMPLETENESS: Is any key information missing from the retrieval? Rate as Good/Poor.
        
        Provide your assessment in JSON format:
        {{
            "retrieval_relevance": "Good/Poor",
            "factual_consistency": "Good/Poor", 
            "completeness": "Good/Poor",
            "overall_rating": "Good/Poor",
            "issues_found": ["list of specific issues or empty list if none"],
            "recommendation": "proceed" or "corrective_action_needed"
        }}
        
        Be strict in your assessment. If any aspect is Poor, recommend corrective_action_needed.
        """)
        
        docs_text = "\n\n".join([f"Doc {i+1}: {doc.page_content}" for i, doc in enumerate(state.retrieved_docs)])
        
        chain = assessment_prompt | self.llm | StrOutputParser()
        assessment_text = chain.invoke({
            "query": state.query,
            "documents": docs_text,
            "draft_answer": state.draft_answer
        })
        
        try:
            assessment = json.loads(assessment_text)
        except:
            # Fallback if JSON parsing fails
            assessment = {
                "retrieval_relevance": "Poor",
                "factual_consistency": "Poor",
                "completeness": "Poor",
                "overall_rating": "Poor",
                "issues_found": ["JSON parsing failed"],
                "recommendation": "corrective_action_needed"
            }
            
        return {"assessment": assessment}
    
    def perform_web_search(self, state: CRAGState) -> Dict[str, Any]:
        """Perform web search to get additional information"""
        try:
            search_results = self.tavily_client.search(
                query=state.query,
                max_results=3
            )
            return {"web_search_results": search_results.get("results", [])}
        except Exception as e:
            print(f"Web search failed: {e}")
            return {"web_search_results": []}
    
    def decide_corrective_action(self, state: CRAGState) -> str:
        """Decide what corrective action to take"""
        assessment = state.assessment
        
        if assessment.get("overall_rating") == "Good":
            return "finalize"
        else:
            issues = assessment.get("issues_found", [])
            if any("completeness" in issue.lower() or "missing" in issue.lower() for issue in issues):
                return "web_search"
            else:
                return "regenerate"
    
    def regenerate_answer(self, state: CRAGState) -> Dict[str, Any]:
        """Regenerate answer with improved context"""
        
        # Combine original docs and web search results
        all_context = []
        
        # Add original documents
        for doc in state.retrieved_docs:
            all_context.append(f"Retrieved Document: {doc.page_content}")
            
        # Add web search results if available
        for result in state.web_search_results:
            all_context.append(f"Web Result: {result.get('content', '')}")
            
        context_text = "\n\n".join(all_context)
        
        regeneration_prompt = ChatPromptTemplate.from_template("""
        Based on the following comprehensive context (including both retrieved documents and web search results), 
        provide a complete and accurate answer to the user's question.
        
        COMPREHENSIVE CONTEXT:
        {context}
        
        QUESTION: {query}
        
        Previous assessment identified these issues: {issues}
        
        Please ensure your answer addresses all identified issues and provides a complete, accurate response.
        
        IMPROVED ANSWER:
        """)
        
        chain = regeneration_prompt | self.llm | StrOutputParser()
        improved_answer = chain.invoke({
            "context": context_text,
            "query": state.query,
            "issues": state.assessment.get("issues_found", [])
        })
        
        return {"final_answer": improved_answer}
    
    def finalize_answer(self, state: CRAGState) -> Dict[str, Any]:
        """Finalize the answer (use draft answer if good)"""
        if state.final_answer:
            return {"final_answer": state.final_answer}
        else:
            return {"final_answer": state.draft_answer}

# Build the LangGraph workflow
def build_crag_workflow(standard_rag, corrective_agent):
    workflow = StateGraph(CRAGState)
    
    # Define nodes
    def retrieve_node(state: CRAGState):
        docs = standard_rag.retrieve(state.query)
        return {"retrieved_docs": docs}
    
    def generate_node(state: CRAGState):
        answer = standard_rag.generate(state.query, state.retrieved_docs)
        return {"draft_answer": answer}
    
    def assess_node(state: CRAGState):
        return corrective_agent.assess_quality(state)
    
    def web_search_node(state: CRAGState):
        return corrective_agent.perform_web_search(state)
    
    def regenerate_node(state: CRAGState):
        return corrective_agent.regenerate_answer(state)
    
    def finalize_node(state: CRAGState):
        return corrective_agent.finalize_answer(state)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("assess", assess_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("regenerate", regenerate_node)
    workflow.add_node("finalize", finalize_node)
    
    # Define edges and conditional logic
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "assess")
    
    def after_assess(state: CRAGState):
        action = corrective_agent.decide_corrective_action(state)
        if action == "finalize":
            return "finalize"
        elif action == "web_search":
            return "web_search"
        else:
            return "regenerate"
    
    workflow.add_conditional_edges(
        "assess",
        after_assess,
        {
            "finalize": "finalize",
            "web_search": "web_search", 
            "regenerate": "regenerate"
        }
    )
    
    workflow.add_edge("web_search", "regenerate")
    workflow.add_edge("regenerate", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow.compile()

# Main CRAG System
class CRAGSystem:
    def __init__(self):
        self.llm, self.vector_store, self.tavily_client = initialize_components()
        self.standard_rag = StandardRAG(self.vector_store, self.llm)
        self.corrective_agent = CorrectiveAgent(self.llm, self.tavily_client)
        self.workflow = build_crag_workflow(self.standard_rag, self.corrective_agent)
    
    def query(self, question: str) -> Dict[str, Any]:
        """Execute the full CRAG pipeline"""
        initial_state = CRAGState(query=question)
        
        # Execute the workflow
        final_state = self.workflow.invoke(initial_state)
        
        return {
            "query": question,
            "retrieved_documents": final_state.retrieved_docs,
            "draft_answer": final_state.draft_answer,
            "assessment": final_state.assessment,
            "web_search_used": len(final_state.web_search_results) > 0,
            "web_search_results": final_state.web_search_results,
            "final_answer": final_state.final_answer,
            "corrective_action_taken": final_state.assessment.get("recommendation") == "corrective_action_needed"
        }

# Usage example
if __name__ == "__main__":
    # Initialize the CRAG system
    crag_system = CRAGSystem()
    
    # Example query
    question = "What are the latest developments in quantum computing for 2024?"
    
    # Get response
    result = crag_system.query(question)
    
    print("=== CRAG SYSTEM RESULTS ===")
    print(f"Query: {result['query']}")
    print(f"\nAssessment: {result['assessment']}")
    print(f"\nCorrective Action Taken: {result['corrective_action_taken']}")
    print(f"\nWeb Search Used: {result['web_search_used']}")
    print(f"\nFinal Answer: {result['final_answer']}")
    
    # Print some details
    print(f"\nRetrieved {len(result['retrieved_documents'])} documents")
    if result['web_search_used']:
        print(f"Added {len(result['web_search_results'])} web search results")