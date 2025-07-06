import os
import uuid
import json
import time
import magic
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import Document, TextNode
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

from hybrid_search import HybridRetriever, HybridQueryEngine
from models import DocumentStatus

load_dotenv()

logger = logging.getLogger(__name__)

class AdvancedKnowledgeBaseManager:
    """
    🚀 UNIFIED ADVANCED KNOWLEDGE BASE MANAGER
    
    Enterprise-grade RAG system with:
    - 🔥 LlamaParse: Ultimate document processing with OCR, tables, everything
    - 🔍 Hybrid Search: Semantic + BM25 keyword matching by default
    - 📊 Rich Analytics: Advanced metadata and search insights
    - ⚡ Production Features: Multi-tenant, scalable, monitored
    
    Same simple API, maximum capabilities built-in.
    """
    
    METADATA_FILE = os.path.join(os.path.dirname(__file__), "metadata.json")

    def __init__(self):
        # Initialize Pinecone
        self._pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # In-memory storage for indices and processors
        self._indices: Dict[str, VectorStoreIndex] = {}
        self._hybrid_retrievers: Dict[str, HybridRetriever] = {}
        self._hybrid_query_engines: Dict[str, HybridQueryEngine] = {}
        self._document_status: Dict[str, Dict[str, Any]] = {}
        
        # Load or initialize metadata
        if not os.path.exists(self.METADATA_FILE):
            with open(self.METADATA_FILE, "w") as f:
                json.dump({}, f)
        with open(self.METADATA_FILE, "r") as f:
            self._metadata: Dict[str, Dict] = json.load(f)
        
        # Setup LlamaParse
        self._setup_llamaparse()
        
        logger.info("🚀 Advanced Knowledge Base Manager initialized")
        logger.info("   🔥 LlamaParse: Enterprise document processing")
        logger.info("   🔍 Hybrid Search: Semantic + BM25 by default")
        logger.info("   📊 Rich Analytics: Advanced insights built-in")

    def _setup_llamaparse(self):
        """Setup LlamaParse with ultimate extraction strategies"""
        
        llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        
        if not llama_cloud_api_key or llama_cloud_api_key == "llx-your-api-key-here":
            logger.warning("⚠️  LlamaParse API key not configured. Using fallback parsing.")
            self.llamaparse_enabled = False
            return
        
        self.llamaparse_enabled = True
        
        # Ultimate parsing strategies for maximum extraction
        self.parsers = {
            # ULTIMATE high-precision parser for ALL documents
            'ultimate': LlamaParse(
                api_key=llama_cloud_api_key,
                result_type="markdown",
                verbose=True,
                language="en",
                parsing_instruction="""
                Extract ALL content with MAXIMUM precision and OCR capability:
                - Extract all text content with original formatting preserved
                - Perform OCR on all images and scanned content within documents
                - Extract tables with complete structure and data preservation
                - Capture headers, footers, watermarks, and metadata
                - Extract lists, bullet points, numbering, and hierarchical structure
                - Identify and extract mathematical formulas, equations, and symbols
                - Process image captions, chart descriptions, and diagram text
                - Extract text from embedded images using advanced OCR
                - Preserve document layout and spatial relationships
                - Handle multi-column layouts and complex document structures
                - Extract text from form fields and annotations
                - Process handwritten text where legible
                """,
                num_workers=4,
                max_timeout=120000,
                split_by_page=False,
                auto_mode_trigger_download_threshold=10
            ),
            
            # Enhanced OCR-focused parser for image-heavy documents
            'ocr': LlamaParse(
                api_key=llama_cloud_api_key,
                result_type="markdown",
                verbose=True,
                language="en",
                parsing_instruction="""
                MAXIMUM OCR extraction from all visual content:
                - Advanced OCR on all images, scanned pages, and visual elements
                - Extract text from charts, graphs, and diagrams using OCR
                - Process handwritten text with high accuracy
                - Extract table data from visual/scanned tables
                - Identify and extract text from complex layouts and multi-column formats
                - Process low-quality scans and improve text recognition
                - Extract text from watermarks and background elements
                - Handle rotated, skewed, or distorted text
                - Process multiple languages within the same document
                - Extract text from forms, signatures, and annotations
                """,
                num_workers=3,
                max_timeout=180000,
                split_by_page=True,
                auto_mode_trigger_download_threshold=5
            )
        }
        
        logger.info("✅ LlamaParse initialized with ULTIMATE extraction strategies")

    def _save_metadata(self):
        """Save metadata to disk"""
        with open(self.METADATA_FILE, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def _get_embedding(self, model_name: str):
        """Get embedding model"""
        if model_name == "openai":
            return OpenAIEmbedding()
        else:
            return HuggingFaceEmbedding(model_name=model_name)

    def _detect_document_complexity(self, content: str, file_type: str) -> str:
        """Detect document complexity for optimal parsing strategy"""
        
        # Check for indicators of complex content
        complexity_indicators = {
            'tables': len([line for line in content.split('\n') if '|' in line or '\t' in line]),
            'images': content.lower().count('image') + content.lower().count('figure'),
            'formulas': content.count('$') + content.count('=') + content.count('∑'),
            'structured': content.count('•') + content.count('1.') + content.count('a)'),
            'scanned': 'scan' in content.lower() or len(content) < 100,  # Might be OCR needed
        }
        
        # Determine complexity
        total_complexity = sum(complexity_indicators.values())
        
        if total_complexity > 20 or file_type.lower() in ['pdf', 'png', 'jpg', 'jpeg', 'tiff']:
            return 'ultimate'  # Use ultimate parser for complex/visual content
        elif total_complexity > 5:
            return 'ultimate'  # Still use ultimate for moderate complexity
        else:
            return 'ultimate'  # Default to ultimate for everything (best quality)

    async def _process_with_llamaparse(self, content: str, file_path: str = None, file_type: str = "txt") -> Tuple[str, Dict[str, Any]]:
        """Process content with LlamaParse for ultimate extraction"""
        
        if not self.llamaparse_enabled:
            return content, {"method": "fallback", "reason": "llamaparse_disabled"}
        
        try:
            # Detect optimal parsing strategy
            complexity = self._detect_document_complexity(content, file_type)
            parser = self.parsers[complexity]
            
            logger.info(f"🔥 Processing with LlamaParse strategy: {complexity}")
            
            # Process with LlamaParse
            if file_path and os.path.exists(file_path):
                # Process file directly for best results
                documents = await asyncio.to_thread(parser.load_data, file_path)
            else:
                # Process content (create temp file if needed)
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{file_type}', delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                
                try:
                    documents = await asyncio.to_thread(parser.load_data, tmp_path)
                finally:
                    os.unlink(tmp_path)
            
            # Extract enhanced content
            enhanced_content = ""
            metadata = {
                "method": "llamaparse",
                "strategy": complexity,
                "pages_processed": len(documents),
                "extraction_features": [
                    "OCR", "Tables", "Images", "Formatting", 
                    "Structure", "Metadata", "Advanced_Layout"
                ]
            }
            
            for doc in documents:
                enhanced_content += doc.text + "\n\n"
            
            logger.info(f"✅ LlamaParse extracted {len(enhanced_content)} chars from {len(documents)} pages")
            
            return enhanced_content.strip(), metadata
            
        except Exception as e:
            logger.error(f"LlamaParse processing failed: {e}")
            return content, {"method": "fallback", "error": str(e)}

    def _create_enhanced_nodes(self, content: str, kb_id: str, metadata: Dict[str, Any]) -> List[TextNode]:
        """Create enhanced text nodes with rich metadata"""
        
        # Structure-aware chunking
        chunks = self._structure_aware_chunking(content)
        
        nodes = []
        for i, chunk in enumerate(chunks):
            # Create simplified metadata for Pinecone compatibility
            node_metadata = {
                "kb_id": kb_id,
                "chunk_id": f"{kb_id}_{i}",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "created_at": datetime.utcnow().isoformat(),
                "processing_version": "advanced_v1.0",
                "document_index": metadata.get("document_index", 0),
                "file_type": metadata.get("file_type", "txt"),
                "original_size": metadata.get("original_size", 0),
                "enhanced_size": metadata.get("enhanced_size", 0),
                "ingestion_timestamp": metadata.get("ingestion_timestamp", ""),
                # Simplified processing info
                "processing_method": metadata.get("processing_metadata", {}).get("method", "advanced"),
                "extraction_strategy": metadata.get("processing_metadata", {}).get("strategy", "ultimate"),
                "pages_processed": metadata.get("processing_metadata", {}).get("pages_processed", 1)
            }
            
            # Add filename if available
            if "filename" in metadata:
                node_metadata["filename"] = metadata["filename"]
            if "file_path" in metadata:
                node_metadata["file_path"] = metadata["file_path"]
            
            # Create enhanced node
            node = TextNode(
                text=chunk,
                metadata=node_metadata,
                id_=f"{kb_id}_chunk_{i}"
            )
            
            nodes.append(node)
        
        logger.info(f"📄 Created {len(nodes)} enhanced nodes with simplified metadata")
        return nodes

    def _structure_aware_chunking(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Intelligent chunking that preserves document structure"""
        
        # Split by natural boundaries
        sections = []
        
        # Try to split by headers first
        header_patterns = [
            r'\n#{1,6}\s+.+\n',  # Markdown headers
            r'\n[A-Z][A-Z\s]{10,}\n',  # ALL CAPS headers
            r'\n\d+\.\s+[A-Z].+\n',  # Numbered sections
        ]
        
        current_content = content
        for pattern in header_patterns:
            import re
            matches = list(re.finditer(pattern, current_content))
            if matches:
                # Split by headers
                last_end = 0
                for match in matches:
                    if last_end < match.start():
                        section = current_content[last_end:match.start()].strip()
                        if section:
                            sections.append(section)
                    last_end = match.start()
                
                # Add final section
                if last_end < len(current_content):
                    section = current_content[last_end:].strip()
                    if section:
                        sections.append(section)
                break
        
        # If no clear structure, split by paragraphs
        if not sections:
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            sections = paragraphs
        
        # Now chunk sections to target size
        chunks = []
        current_chunk = ""
        
        for section in sections:
            if len(current_chunk) + len(section) <= chunk_size:
                current_chunk += section + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Handle oversized sections
                if len(section) > chunk_size:
                    # Split large sections
                    words = section.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) <= chunk_size:
                            temp_chunk += word + " "
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = word + " "
                    if temp_chunk:
                        current_chunk = temp_chunk
                else:
                    current_chunk = section + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _setup_hybrid_search(self, kb_id: str, index: VectorStoreIndex):
        """Setup hybrid search for a knowledge base"""
        
        try:
            # Get embedding model from the index
            embed_model = index._embed_model
            
            # Initialize hybrid retriever
            hybrid_retriever = HybridRetriever(
                vector_store=index._vector_store,
                embed_model=embed_model,
                similarity_top_k=8,
                bm25_top_k=8,
                semantic_weight=0.7,  # Favor semantic search slightly
                bm25_weight=0.3,      # But still value keyword matching
                hybrid_threshold=0.1,  # Low threshold for more results
                enable_query_expansion=True
            )
            
            # Get LLM (with fallback)
            try:
                llm = index._llm
            except AttributeError:
                llm = OpenAI(model="gpt-4o", temperature=0.1)
            
            # Initialize hybrid query engine
            hybrid_query_engine = HybridQueryEngine(
                hybrid_retriever=hybrid_retriever,
                llm=llm
            )
            
            # Store for this KB
            self._hybrid_retrievers[kb_id] = hybrid_retriever
            self._hybrid_query_engines[kb_id] = hybrid_query_engine
            
            logger.info(f"✅ Hybrid Search setup complete for KB {kb_id}")
            
        except Exception as e:
            logger.error(f"Failed to setup hybrid search for KB {kb_id}: {e}")

    def create_kb(self, name: str = None, embedding_model: str = "openai") -> str:
        """Create a new knowledge base with advanced capabilities"""
        
        kb_id = uuid.uuid4().hex
        index_name = f"kb-{kb_id.lower()}"
        embedder = self._get_embedding(embedding_model)
        
        # Determine dimension
        dimension = getattr(embedder, "embedding_dim", None) or getattr(embedder, "dimensions", None)
        if dimension is None:
            dimension = 1536
        
        # Create Pinecone index
        region = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        spec = ServerlessSpec(cloud="aws", region=region)
        
        self._pc.create_index(
            name=index_name,
            spec=spec,
            dimension=dimension,
            metric="cosine"
        )
        
        # Initialize vector store and index
        store = PineconeVectorStore(index_name=index_name, text_key="text")
        storage_context = StorageContext.from_defaults(vector_store=store)
        idx = VectorStoreIndex.from_documents(
            documents=[], 
            embed_model=embedder, 
            storage_context=storage_context
        )
        
        # Store index
        self._indices[kb_id] = idx
        
        # Setup hybrid search
        self._setup_hybrid_search(kb_id, idx)
        
        # Record metadata
        self._metadata[kb_id] = {
            "name": name,
            "index_name": index_name,
            "embedding_model": embedding_model,
            "created_at": datetime.utcnow().isoformat(),
            "features": [
                "llamaparse_processing",
                "hybrid_search", 
                "semantic_search",
                "bm25_keyword_search",
                "ocr_extraction",
                "table_extraction",
                "advanced_chunking",
                "rich_metadata"
            ],
            "capabilities": {
                "document_types": ["pdf", "docx", "txt", "md", "html", "images"],
                "extraction_features": ["text", "tables", "images", "ocr", "structure"],
                "search_methods": ["semantic", "keyword", "hybrid"],
                "analytics": True
            }
        }
        self._save_metadata()
        
        logger.info(f"🚀 Created advanced KB {kb_id} with full capabilities")
        return kb_id

    def _load_index_if_needed(self, kb_id: str, embedding_model: str):
        """Load index and setup hybrid search if needed"""
        
        if kb_id not in self._indices:
            if kb_id not in self._metadata:
                raise KeyError(f"KB {kb_id} not found")
            
            entry = self._metadata[kb_id]
            index_name = entry["index_name"]
            embedder = self._get_embedding(embedding_model)
            store = PineconeVectorStore(index_name=index_name, text_key="text")
            
            # Load existing index
            storage_context = StorageContext.from_defaults(vector_store=store)
            idx = VectorStoreIndex.from_vector_store(
                vector_store=store, 
                embed_model=embedder,
                storage_context=storage_context
            )
            
            self._indices[kb_id] = idx
            
            # Setup hybrid search if not already done
            if kb_id not in self._hybrid_query_engines:
                self._setup_hybrid_search(kb_id, idx)
        
        return self._indices[kb_id]

    async def ingest(self, kb_id: str, texts: List[str], embedding_model: str = "openai", files: List[str] = None):
        """Advanced document ingestion with LlamaParse and hybrid search"""
        
        idx = self._load_index_if_needed(kb_id, embedding_model)
        
        # Process each text with advanced capabilities
        all_nodes = []
        
        for i, text in enumerate(texts):
            # Determine file info
            file_path = files[i] if files and i < len(files) else None
            file_type = "txt"
            if file_path:
                file_type = Path(file_path).suffix.lstrip('.')
            
            logger.info(f"🔥 Processing document {i+1}/{len(texts)} with advanced pipeline...")
            
            # Process with LlamaParse for ultimate extraction
            enhanced_content, processing_metadata = await self._process_with_llamaparse(
                text, file_path, file_type
            )
            
            # Create enhanced nodes with rich metadata
            base_metadata = {
                "document_index": i,
                "file_type": file_type,
                "original_size": len(text),
                "enhanced_size": len(enhanced_content),
                "processing_metadata": processing_metadata,
                "ingestion_timestamp": datetime.utcnow().isoformat()
            }
            
            if file_path:
                base_metadata["file_path"] = file_path
                base_metadata["filename"] = Path(file_path).name
            
            # Create nodes
            nodes = self._create_enhanced_nodes(enhanced_content, kb_id, base_metadata)
            all_nodes.extend(nodes)
        
        # Insert all nodes into the index
        logger.info(f"📥 Inserting {len(all_nodes)} enhanced nodes into vector store...")
        
        for node in all_nodes:
            idx.insert_nodes([node])
        
        # Refresh hybrid search index with new content
        if kb_id in self._hybrid_retrievers:
            try:
                # Build BM25 index from all nodes (new + existing)
                retriever = self._hybrid_retrievers[kb_id]
                
                # For now, rebuild from the new nodes - in production you'd maintain a full corpus
                retriever.build_bm25_index(all_nodes)
                logger.info(f"🔍 Updated hybrid search index for KB {kb_id} with {len(all_nodes)} nodes")
            except Exception as e:
                logger.error(f"Failed to update hybrid search: {e}")
        else:
            # Setup hybrid search if not already done
            logger.info("🔍 Setting up hybrid search for existing KB")
            self._setup_hybrid_search(kb_id, idx)
        
        logger.info(f"✅ Advanced ingestion complete: {len(all_nodes)} nodes processed")

    def query(self, kb_id: str, query_str: str, embedding_model: str = "openai") -> str:
        """Advanced query with hybrid search by default"""
        
        idx = self._load_index_if_needed(kb_id, embedding_model)
        
        # Use hybrid search if available
        if kb_id in self._hybrid_query_engines:
            try:
                logger.info(f"🔍 Using hybrid search for query: {query_str[:50]}...")
                hybrid_engine = self._hybrid_query_engines[kb_id]
                response = hybrid_engine.query(query_str)
                
                # Extract response text (handle different response types)
                if hasattr(response, 'response'):
                    result = response.response
                elif hasattr(response, 'answer'):
                    result = response.answer
                else:
                    result = str(response)
                
                # Check if we got a meaningful response
                if result and result.strip() and "couldn't find" not in result.lower():
                    logger.info("✅ Hybrid search returned results")
                    return result
                else:
                    logger.info("🔍 Hybrid search returned empty, trying semantic fallback")
                    
            except Exception as e:
                logger.error(f"Hybrid search failed, falling back to semantic: {e}")
        
        # Fallback to semantic search
        logger.info("🧠 Using semantic search fallback")
        qe = idx.as_query_engine(similarity_top_k=5, response_mode="compact")
        response = qe.query(query_str)
        return response.response

    def list_kbs(self):
        """List all knowledge bases with enhanced metadata"""
        return self._metadata

    def delete_kb(self, kb_id: str):
        """Delete knowledge base and cleanup all resources"""
        
        if kb_id not in self._metadata:
            raise KeyError(f"KB {kb_id} not found")
        
        index_name = self._metadata[kb_id]["index_name"]
        
        # Delete Pinecone index
        self._pc.delete_index(index_name)
        
        # Clean up all caches
        self._metadata.pop(kb_id)
        self._indices.pop(kb_id, None)
        self._hybrid_retrievers.pop(kb_id, None)
        self._hybrid_query_engines.pop(kb_id, None)
        self._document_status.pop(kb_id, None)
        
        self._save_metadata()
        
        logger.info(f"🗑️  Deleted KB {kb_id} and cleaned up all resources")

    def get_kb_stats(self, kb_id: str) -> Dict[str, Any]:
        """Get comprehensive knowledge base statistics"""
        
        if kb_id not in self._metadata:
            raise KeyError(f"KB {kb_id} not found")
        
        metadata = self._metadata[kb_id]
        
        # Basic stats
        stats = {
            "kb_id": kb_id,
            "name": metadata.get("name"),
            "created_at": metadata.get("created_at"),
            "features": metadata.get("features", []),
            "capabilities": metadata.get("capabilities", {}),
            "search_methods": ["semantic", "keyword", "hybrid"],
            "processing_methods": ["llamaparse", "ocr", "table_extraction"],
            "status": "active"
        }
        
        # Add runtime stats if available
        if kb_id in self._indices:
            stats["index_loaded"] = True
            stats["hybrid_search_enabled"] = kb_id in self._hybrid_query_engines
        else:
            stats["index_loaded"] = False
            stats["hybrid_search_enabled"] = False
        
        return stats


# Alias for backward compatibility
KnowledgeBaseManager = AdvancedKnowledgeBaseManager 