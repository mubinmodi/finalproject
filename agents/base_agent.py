"""
Base Agent Class with Gemini API and FAISS Vector Store

Provides common functionality for all analysis agents including:
- FAISS vector store for RAG (like rag_chat.py)
- Gemini LLM interface
- Document loading and retrieval
- Citation tracking
"""

import json
import os
import gc
from pathlib import Path
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
import numpy as np

# Core imports
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from utils import config, get_logger

logger = get_logger("agents")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
    logger.info(f"✅ Gemini API configured with model: {GEMINI_MODEL}")
else:
    GEMINI_AVAILABLE = False
    logger.warning("⚠️  Gemini API key not found in .env")


class BaseAgent(ABC):
    """Base class for all analysis agents with Gemini and FAISS."""
    
    def __init__(
        self,
        agent_name: str,
        model: Optional[str] = None,
        temperature: float = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize base agent.
        
        Args:
            agent_name: Name of the agent
            model: LLM model name (defaults to GEMINI_MODEL)
            temperature: LLM temperature (defaults to config)
            embedding_model: Sentence transformer model for embeddings
        """
        self.agent_name = agent_name
        self.model_name = model or GEMINI_MODEL
        self.temperature = temperature if temperature is not None else config.agent.temperature
        
        logger.info(f"Initializing {agent_name}")
        
        # Initialize Gemini
        if GEMINI_AVAILABLE:
            try:
                self.llm = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config={
                        "temperature": self.temperature,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": config.agent.max_tokens,
                    }
                )
                logger.info(f"✅ {agent_name} initialized with {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self.llm = None
        else:
            self.llm = None
            logger.warning(f"⚠️  {agent_name} initialized in fallback mode")
        
        # Initialize FAISS components (like rag_chat.py)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunk_mapping = {}
        
        logger.info(f"FAISS index initialized with dimension: {self.dimension}")
    
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
        force_rebuild: bool = False
    ) -> None:
        """
        Create or load FAISS vector store for a document (like rag_chat.py).
        
        Args:
            doc_id: Document identifier
            force_rebuild: Force rebuild even if cache exists
        """
        # Paths for FAISS index and chunk mapping
        vector_dir = config.paths.data_dir / "vector_stores" / doc_id
        vector_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = vector_dir / "vector_index.faiss"
        mapping_path = vector_dir / "chunk_mapping.json"
        
        # Try to load existing index
        if not force_rebuild and index_path.exists() and mapping_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                logger.info(f"Loaded existing FAISS index from {index_path}")
                
                with open(mapping_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                self.chunk_mapping = {int(k): v for k, v in loaded.items()}
                logger.info(f"Loaded chunk mapping with {len(self.chunk_mapping)} entries")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached index: {e}. Rebuilding...")
        
        # Build new index
        logger.info("Building new FAISS index...")
        chunks = self.load_document(doc_id)
        
        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Create embeddings in batches (memory efficient like rag_chat.py)
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            embeddings = self.embedding_model.encode(batch, normalize_embeddings=True)
            all_embeddings.append(embeddings)
            gc.collect()
        
        # Combine all embeddings
        final_embeddings = np.vstack(all_embeddings)
        
        # Add to FAISS index
        self.index.add(final_embeddings)
        logger.info(f"Added {len(texts)} vectors to FAISS index")
        
        # Create chunk mapping
        self.chunk_mapping = {i: chunk for i, chunk in enumerate(chunks)}
        
        # Save index and mapping
        faiss.write_index(self.index, str(index_path))
        logger.info(f"FAISS index saved to {index_path}")
        
        with open(mapping_path, "w", encoding="utf-8") as f:
            # Save only essential chunk data
            mapping_to_save = {
                i: {
                    'text': chunk['text'],
                    'page': chunk.get('page', 0),
                    'section': chunk.get('section'),
                    'chunk_id': chunk['chunk_id']
                }
                for i, chunk in self.chunk_mapping.items()
            }
            json.dump(mapping_to_save, f, ensure_ascii=False, indent=2)
        logger.info(f"Chunk mapping saved to {mapping_path}")
        
        # Clean up memory
        del all_embeddings
        del final_embeddings
        gc.collect()
    
    def retrieve_relevant_chunks(
        self,
        query: str = None,
        k: int = 5,
        doc_id: str = None,  # For backward compatibility
        vector_store: Any = None  # For backward compatibility (ignored)
    ) -> List[Dict]:
        """
        Retrieve relevant chunks using FAISS similarity search.
        
        Args:
            query: Search query
            k: Number of chunks to retrieve
            doc_id: Document identifier (for backward compatibility, ignored)
            vector_store: Vector store (for backward compatibility, ignored)
        
        Returns:
            List of relevant chunk dictionaries
        """
        if self.index.ntotal == 0:
            logger.warning("No vectors in FAISS index")
            return []
        
        if not query:
            logger.error("Query is required")
            return []
        
        try:
            # Embed query
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
            
            # Search FAISS index
            k = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_embedding, k)
            
            # Retrieve chunks
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx in self.chunk_mapping:
                    chunk = self.chunk_mapping[idx].copy()
                    chunk['relevance_score'] = float(1.0 / (1.0 + distance))  # Convert distance to similarity
                    results.append(chunk)
            
            logger.info(f"Retrieved {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def query_llm(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_retries: int = 3
    ) -> str:
        """
        Query Gemini LLM with prompt and optional context.
        
        Args:
            prompt: User prompt
            context: Optional context to prepend
            max_retries: Maximum retry attempts
        
        Returns:
            LLM response text
        """
        if not GEMINI_AVAILABLE or not self.llm:
            return f"""[Agent: {self.agent_name}]

LLM not available. This is a placeholder response.

Your query: 
{prompt}

{f'Context: {context[:500]}...' if context else ''}

To enable full functionality, please:
1. Install required packages: pip install google-generativeai sentence-transformers faiss-cpu
2. Set GEMINI_API_KEY in .env file"""
        
        # Construct full prompt
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        
        # Query Gemini with retries
        for attempt in range(max_retries):
            try:
                response = self.llm.generate_content(full_prompt)
                return response.text
            except Exception as e:
                logger.warning(f"Gemini query failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
        
        return "Error: Failed to get response from Gemini"
    
    def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None
    ) -> str:
        """
        Generate response using Gemini LLM (alias for query_llm for backward compatibility).
        
        Args:
            prompt: User prompt
            context: Optional context to include
        
        Returns:
            Generated response
        """
        return self.query_llm(prompt, context)
    
    def save_analysis(
        self,
        doc_id: str,
        analysis: Dict[str, Any],
        filename: str
    ) -> Path:
        """
        Save analysis results to JSON file.
        
        Args:
            doc_id: Document identifier
            analysis: Analysis results dictionary
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        output_dir = config.paths.final_dir / doc_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {self.agent_name} analysis to {output_path}")
        return output_path
    
    @abstractmethod
    def analyze(self, doc_id: str, **kwargs) -> Dict[str, Any]:
        """
        Perform analysis on document. Must be implemented by subclasses.
        
        Args:
            doc_id: Document identifier
            **kwargs: Additional analysis parameters
        
        Returns:
            Analysis results dictionary
        """
        pass
    
    def cleanup(self):
        """Clean up resources and memory."""
        if hasattr(self, 'index'):
            del self.index
        if hasattr(self, 'chunk_mapping'):
            del self.chunk_mapping
        if hasattr(self, 'embedding_model'):
            del self.embedding_model
        gc.collect()
        logger.info(f"{self.agent_name} cleaned up")
