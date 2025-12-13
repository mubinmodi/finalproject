"""
Base Agent Class

Provides common functionality for all analysis agents including:
- Vector store (ChromaDB) for RAG
- LLM interface (OpenAI)
- Document loading and retrieval
- Citation tracking
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
from utils import config, get_logger

logger = get_logger("agents")

# Try to import LangChain and related libraries
try:
    from langchain.chat_models import ChatOpenAI
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.schema import Document
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not available. Agents will use fallback mode.")
    LANGCHAIN_AVAILABLE = False


class BaseAgent(ABC):
    """Base class for all analysis agents."""
    
    def __init__(
        self,
        agent_name: str,
        model: Optional[str] = None,
        temperature: float = None
    ):
        """
        Initialize base agent.
        
        Args:
            agent_name: Name of the agent
            model: LLM model name (defaults to config)
            temperature: LLM temperature (defaults to config)
        """
        self.agent_name = agent_name
        self.model_name = model or config.agent.model
        self.temperature = temperature if temperature is not None else config.agent.temperature
        
        logger.info(f"Initializing {agent_name}")
        
        if LANGCHAIN_AVAILABLE:
            # Initialize LLM
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=config.agent.max_tokens,
                openai_api_key=config.agent.openai_api_key
            )
            
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(
                model=config.agent.embedding_model,
                openai_api_key=config.agent.openai_api_key
            )
            
            logger.info(f"✅ {agent_name} initialized with {self.model_name}")
        else:
            self.llm = None
            self.embeddings = None
            logger.warning(f"⚠️  {agent_name} initialized in fallback mode")
    
    def load_document(self, doc_id: str) -> List[Dict]:
        """
        Load processed document chunks.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            List of chunk dictionaries
        """
        chunks_path = config.paths.final_dir / doc_id / "chunks.jsonl"
        
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks not found for {doc_id}")
        
        chunks = []
        with open(chunks_path, 'r') as f:
            for line in f:
                chunks.append(json.loads(line.strip()))
        
        logger.info(f"Loaded {len(chunks)} chunks from {doc_id}")
        return chunks
    
    def create_vector_store(
        self,
        doc_id: str,
        persist_directory: Optional[str] = None
    ) -> Any:
        """
        Create or load vector store for a document.
        
        Args:
            doc_id: Document identifier
            persist_directory: Optional directory to persist vector store
        
        Returns:
            Vector store instance
        """
        if not LANGCHAIN_AVAILABLE:
            logger.error("Vector store not available without LangChain")
            return None
        
        # Load chunks
        chunks = self.load_document(doc_id)
        
        # Convert chunks to LangChain documents
        documents = []
        for chunk in chunks:
            metadata = {
                'doc_id': chunk['doc_id'],
                'chunk_id': chunk['chunk_id'],
                'page': chunk['page'],
                'item': chunk.get('item'),
                'section': chunk.get('section'),
                'source_path': chunk['source_path']
            }
            
            doc = Document(
                page_content=chunk['text'],
                metadata=metadata
            )
            documents.append(doc)
        
        # Create persist directory
        if persist_directory is None:
            persist_directory = str(config.paths.data_dir / "vector_stores" / doc_id)
        
        # Create vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory,
            collection_name=doc_id
        )
        
        logger.info(f"Created vector store with {len(documents)} documents")
        return vector_store
    
    def retrieve_relevant_chunks(
        self,
        doc_id: str,
        query: str,
        k: int = 5,
        vector_store: Optional[Any] = None
    ) -> List[Dict]:
        """
        Retrieve relevant chunks using semantic search.
        
        Args:
            doc_id: Document identifier
            query: Search query
            k: Number of chunks to retrieve
            vector_store: Optional pre-loaded vector store
        
        Returns:
            List of relevant chunks with scores
        """
        if not LANGCHAIN_AVAILABLE:
            logger.warning("Using fallback retrieval (keyword-based)")
            return self._fallback_retrieval(doc_id, query, k)
        
        # Create or use existing vector store
        if vector_store is None:
            vector_store = self.create_vector_store(doc_id)
        
        # Retrieve relevant documents
        docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
        
        # Convert to chunk format
        results = []
        for doc, score in docs_with_scores:
            result = {
                'text': doc.page_content,
                'metadata': doc.metadata,
                'relevance_score': float(score)
            }
            results.append(result)
        
        logger.debug(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
        return results
    
    def _fallback_retrieval(
        self,
        doc_id: str,
        query: str,
        k: int
    ) -> List[Dict]:
        """
        Fallback keyword-based retrieval when vector store is unavailable.
        
        Args:
            doc_id: Document identifier
            query: Search query
            k: Number of chunks to retrieve
        
        Returns:
            List of relevant chunks
        """
        chunks = self.load_document(doc_id)
        
        # Simple keyword matching
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for chunk in chunks:
            text_words = set(chunk['text'].lower().split())
            # Calculate Jaccard similarity
            intersection = len(query_words & text_words)
            union = len(query_words | text_words)
            score = intersection / union if union > 0 else 0
            
            if score > 0:
                scored_chunks.append({
                    'text': chunk['text'],
                    'metadata': {
                        'doc_id': chunk['doc_id'],
                        'chunk_id': chunk['chunk_id'],
                        'page': chunk['page']
                    },
                    'relevance_score': score
                })
        
        # Sort by score and return top k
        scored_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        return scored_chunks[:k]
    
    def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None
    ) -> str:
        """
        Generate response using LLM.
        
        Args:
            prompt: User prompt
            context: Optional context to include
        
        Returns:
            Generated response
        """
        if not LANGCHAIN_AVAILABLE:
            return self._fallback_response(prompt)
        
        # Construct full prompt with context
        if context:
            full_prompt = f"Context:\n{context}\n\nQuery:\n{prompt}"
        else:
            full_prompt = prompt
        
        # Generate response
        response = self.llm.predict(full_prompt)
        return response
    
    def _fallback_response(self, prompt: str) -> str:
        """Fallback response when LLM is unavailable."""
        return (
            f"[Agent: {self.agent_name}]\n\n"
            f"LLM not available. This is a placeholder response.\n\n"
            f"Your query: {prompt}\n\n"
            f"To enable full functionality, please:\n"
            f"1. Install required packages: pip install langchain openai chromadb\n"
            f"2. Set OPENAI_API_KEY in .env file"
        )
    
    @abstractmethod
    def analyze(self, doc_id: str, **kwargs) -> Dict[str, Any]:
        """
        Perform agent-specific analysis.
        
        Args:
            doc_id: Document identifier
            **kwargs: Agent-specific parameters
        
        Returns:
            Analysis results dictionary
        """
        pass
    
    def save_analysis(
        self,
        doc_id: str,
        analysis: Dict[str, Any],
        output_name: str
    ):
        """
        Save analysis results to JSON file.
        
        Args:
            doc_id: Document identifier
            analysis: Analysis results
            output_name: Output file name (without extension)
        """
        output_dir = config.paths.final_dir / doc_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{output_name}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {self.agent_name} analysis to {output_path}")
