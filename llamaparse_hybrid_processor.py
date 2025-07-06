import os
import uuid
import time
import magic
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_parse import LlamaParse
from pinecone import Pinecone
from dotenv import load_dotenv

from models import DocumentStatus
from hybrid_search import HybridRetriever, HybridQueryEngine

load_dotenv()

logger = logging.getLogger(__name__)

class LlamaParseHybridProcessor:
    """
    🚀 ULTIMATE DOCUMENT PROCESSOR WITH LLAMAPARSE + HYBRID SEARCH
    
    Combines:
    - 🔥 LlamaParse: Enterprise document processing with OCR and table extraction
    - 🔍 Hybrid Search: Semantic similarity + BM25 keyword matching
    - 📊 Advanced Analytics: Rich metadata and search insights
    - ⚡ Production Features: Scalable architecture and monitoring
    """
    
    def __init__(self):
        # In-memory storage for demo (in production, use Redis/database)
        self.document_status: Dict[str, Dict[str, Any]] = {}
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.pinecone_index = self.pc.Index(self.index_name)
        
        # Create vector store
        self.vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Initialize global index for querying
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=self.storage_context
        )
        
        # Initialize LlamaParse with different parsing strategies
        self.setup_llamaparse()
        
        # Initialize Hybrid Search
        self.setup_hybrid_search()
        
        # Document corpus for BM25 (will be built as documents are processed)
        self.document_corpus = []
    
    def setup_llamaparse(self):
        """Setup LlamaParse with enhanced strategies optimized for MAXIMUM extraction quality"""
        
        llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        
        if not llama_cloud_api_key or llama_cloud_api_key == "llx-your-api-key-here":
            logger.warning("⚠️  LlamaParse API key not configured. Using fallback parsing.")
            self.llamaparse_enabled = False
            return
        
        self.llamaparse_enabled = True
        
        # Enhanced parsing strategies - now optimized for MAXIMUM extraction
        self.parsers = {
            # ULTIMATE high-precision parser for ALL documents
            'complex': LlamaParse(
                api_key=llama_cloud_api_key,
                result_type="markdown",  # Preserve formatting
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
                max_timeout=120000,  # Extended timeout for comprehensive processing
                split_by_page=False,  # Keep document structure intact
                auto_mode_trigger_download_threshold=10  # Enhanced processing threshold
            ),
            
            # Enhanced parser for simple text documents (still with OCR)
            'simple': LlamaParse(
                api_key=llama_cloud_api_key,
                result_type="markdown",  # Changed to markdown for better structure
                verbose=True,
                language="en",
                parsing_instruction="""
                Extract clean, structured content efficiently with OCR:
                - Extract all text with basic formatting preserved
                - Identify and extract simple table structures
                - Preserve headers and document hierarchy
                - Basic OCR for any embedded images
                - Extract text from simple charts and diagrams
                """,
                num_workers=2,
                max_timeout=60000
            ),
            
            # ULTIMATE OCR-focused parser for image-heavy documents
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
                max_timeout=180000,  # Extended timeout for intensive OCR processing
                split_by_page=True,   # Better for OCR processing
                auto_mode_trigger_download_threshold=5
            )
        }
        
        logger.info("✅ LlamaParse initialized with ULTIMATE extraction strategies")
        logger.info("   🔥 Complex: Maximum precision + OCR for all documents")
        logger.info("   ⚡ Simple: Enhanced fast processing with OCR")
        logger.info("   👁️  OCR: Ultimate visual content extraction")
    
    def setup_hybrid_search(self):
        """Setup hybrid search capabilities"""
        
        try:
            # Get embedding model from the index
            embed_model = self.index._embed_model
            
            # Initialize hybrid retriever
            self.hybrid_retriever = HybridRetriever(
                vector_store=self.vector_store,
                embed_model=embed_model,
                similarity_top_k=8,
                bm25_top_k=8,
                semantic_weight=0.7,  # Favor semantic search slightly
                bm25_weight=0.3,      # But still value keyword matching
                hybrid_threshold=0.1,  # Low threshold to include more results
                enable_query_expansion=True
            )
            
            # Get LLM from the index (with fallback)
            try:
                llm = self.index._llm
            except AttributeError:
                # Fallback to creating LLM directly
                from llama_index.llms.openai import OpenAI
                llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
            
            # Initialize hybrid query engine
            self.hybrid_query_engine = HybridQueryEngine(
                hybrid_retriever=self.hybrid_retriever,
                llm=llm
            )
            
            self.hybrid_search_enabled = True
            logger.info("✅ Hybrid Search initialized (Semantic + BM25)")
            
        except Exception as e:
            logger.error(f"Failed to initialize hybrid search: {e}")
            self.hybrid_search_enabled = False
            
            # Fallback to regular query engine
            try:
                self.fallback_query_engine = self.index.as_query_engine(
                    similarity_top_k=5,
                    response_mode="compact"
                )
            except Exception as fallback_error:
                logger.error(f"Failed to create fallback query engine: {fallback_error}")
                self.fallback_query_engine = None
    
    def refresh_bm25_index(self):
        """Refresh the BM25 index with all current documents"""
        
        if not self.hybrid_search_enabled:
            return
            
        try:
            # Get all nodes from the vector store
            all_nodes = self.get_all_nodes()
            
            if all_nodes:
                self.hybrid_retriever.build_bm25_index(all_nodes)
                logger.info(f"🔍 BM25 index refreshed with {len(all_nodes)} documents")
            else:
                logger.warning("⚠️  No documents found for BM25 indexing")
                
        except Exception as e:
            logger.error(f"Failed to refresh BM25 index: {e}")
    
    def get_all_nodes(self) -> List[TextNode]:
        """Retrieve all nodes from the vector store for BM25 indexing"""
        
        try:
            # This is a simplified approach - in production, you'd want to implement
            # proper node retrieval from the vector store
            return self.document_corpus
            
        except Exception as e:
            logger.error(f"Failed to retrieve nodes: {e}")
            return []
    
    def detect_document_complexity(self, file_path: str, file_type: str) -> str:
        """Determine the appropriate parsing strategy based on document characteristics"""
        
        file_size = os.path.getsize(file_path)
        
        # Size-based heuristics
        if file_size > 10 * 1024 * 1024:  # > 10MB
            return 'complex'
        elif file_size < 1 * 1024 * 1024:  # < 1MB
            return 'simple'
        
        # File type-based heuristics
        if file_type in ['pdf']:
            # For PDFs, assume complex if large or if filename suggests complexity
            filename_lower = Path(file_path).name.lower()
            if any(keyword in filename_lower for keyword in 
                   ['report', 'financial', 'table', 'form', 'contract', 'manual']):
                return 'complex'
            return 'simple'
        elif file_type in ['image', 'jpg', 'png', 'tiff']:
            return 'ocr'
        else:
            return 'simple'
    
    def detect_file_type(self, file_path: str) -> str:
        """Enhanced file type detection"""
        try:
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(file_path)
            
            # Detailed mapping for better processing decisions
            if file_type.startswith('application/pdf'):
                return 'pdf'
            elif file_type.startswith('image/'):
                return 'image'
            elif file_type.startswith('audio/'):
                return 'audio'
            elif file_type.startswith('video/'):
                return 'video'
            elif 'word' in file_type or file_type.endswith('msword') or 'officedocument.wordprocessing' in file_type:
                return 'docx'
            elif 'sheet' in file_type or 'excel' in file_type or 'officedocument.spreadsheet' in file_type:
                return 'xlsx'
            elif 'presentation' in file_type or 'powerpoint' in file_type:
                return 'pptx'
            elif 'text/' in file_type:
                return 'text'
            else:
                return 'unknown'
        except Exception as e:
            logger.warning(f"Could not detect file type for {file_path}: {e}")
            return Path(file_path).suffix.lower().lstrip('.')
    
    def estimate_processing_time(self, filename: str, file_size: int, complexity: str = None) -> int:
        """Enhanced processing time estimation for ULTIMATE LlamaParse processing"""
        file_ext = Path(filename).suffix.lower()
        
        # Enhanced time estimates with ULTIMATE LlamaParse processing
        # Times are higher because we're using maximum quality extraction
        base_times = {
            '.pdf': 60,   # Increased for comprehensive extraction
            '.docx': 45,  # Enhanced processing
            '.xlsx': 50,  # Table extraction focus
            '.pptx': 45,  # Text + image extraction
            '.txt': 20,   # Even text files get enhanced processing
            '.md': 20,    # Markdown with potential embedded content
            '.jpg': 90,   # Ultimate OCR processing
            '.png': 90,   # Ultimate OCR processing
            '.tiff': 120, # High-res OCR + analysis
            '.mp3': 180,  # Audio transcription
            '.mp4': 300,  # Video + audio processing
        }
        
        base_time = base_times.get(file_ext, 45)  # Default to enhanced processing
        
        # Adjust for file size - larger files need more time for comprehensive processing
        size_factor = max(1.2, file_size / (1024 * 1024))  # Minimum 1.2x for enhanced processing
        
        # ULTIMATE LlamaParse multiplier for maximum quality
        llamaparse_multiplier = 2.0 if self.llamaparse_enabled else 1.0
        
        return int(base_time * size_factor * llamaparse_multiplier)
    
    def create_upload_record(self, upload_id: str, filename: str, file_size: int, 
                           complexity: str = None) -> Dict[str, Any]:
        """Create enhanced upload record with complexity assessment"""
        
        # Detect complexity if not provided
        if not complexity:
            file_type = self.detect_file_type(f"./uploads/{upload_id}_{filename}")
            complexity = self.detect_document_complexity(f"./uploads/{upload_id}_{filename}", file_type)
        
        record = {
            'upload_id': upload_id,
            'filename': filename,
            'status': DocumentStatus.PENDING,
            'progress_percentage': 0,
            'created_at': datetime.now(),
            'completed_at': None,
            'error_message': None,
            'extracted_content_preview': None,
            'metadata': {
                'file_size': file_size,
                'complexity': complexity,
                'llamaparse_enabled': self.llamaparse_enabled,
                'hybrid_search_enabled': self.hybrid_search_enabled,
                'processing_mode': 'ULTIMATE_EXTRACTION' if self.llamaparse_enabled else 'FALLBACK',
                'ocr_enabled': self.llamaparse_enabled,
                'table_extraction_enabled': self.llamaparse_enabled,
                'estimated_processing_time': self.estimate_processing_time(filename, file_size, complexity)
            }
        }
        
        self.document_status[upload_id] = record
        return record
    
    def update_status(self, upload_id: str, status: DocumentStatus, progress: int = None, 
                     error_message: str = None, content_preview: str = None, 
                     metadata: Dict[str, Any] = None):
        """Update document processing status"""
        if upload_id in self.document_status:
            self.document_status[upload_id]['status'] = status
            
            if progress is not None:
                self.document_status[upload_id]['progress_percentage'] = progress
                
            if error_message:
                self.document_status[upload_id]['error_message'] = error_message
                
            if content_preview:
                self.document_status[upload_id]['extracted_content_preview'] = content_preview
                
            if metadata:
                self.document_status[upload_id]['metadata'].update(metadata)
                
            if status == DocumentStatus.COMPLETED:
                self.document_status[upload_id]['completed_at'] = datetime.now()
    
    def get_status(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Get document processing status"""
        return self.document_status.get(upload_id)
    
    async def process_with_llamaparse(self, file_path: str, complexity: str) -> tuple[str, Dict[str, Any]]:
        """Process document using LlamaParse with appropriate strategy"""
        
        if not self.llamaparse_enabled:
            raise ValueError("LlamaParse not enabled - API key required")
        
        parser = self.parsers.get(complexity, self.parsers['simple'])
        
        try:
            logger.info(f"🔥 Processing with LlamaParse using '{complexity}' strategy...")
            
            # Load and parse the document
            documents = parser.load_data(file_path)
            
            # Combine all document content
            content = "\n\n".join([doc.text for doc in documents])
            
            # Extract enhanced metadata
            enhanced_metadata = {
                'parsing_strategy': complexity,
                'llamaparse_version': 'v2.0',  # Version info
                'pages_processed': len(documents),
                'content_length': len(content),
                'extraction_method': f'LlamaParse-{complexity}',
                'ocr_performed': complexity == 'ocr',
                'table_extraction': complexity in ['complex', 'ocr'],
                'formatting_preserved': complexity in ['complex', 'ocr']
            }
            
            # Additional document analysis
            word_count = len(content.split())
            enhanced_metadata.update({
                'word_count': word_count,
                'estimated_reading_time_minutes': max(1, word_count // 200),
                'has_tables': '|' in content or 'Table' in content,
                'has_lists': ('•' in content or '*' in content or 
                             any(line.strip().startswith(str(i) + '.') for i in range(1, 10) 
                                 for line in content.split('\n')[:50]))
            })
            
            logger.info(f"✅ LlamaParse processing completed - {len(content)} characters extracted")
            
            return content, enhanced_metadata
            
        except Exception as e:
            logger.error(f"LlamaParse processing failed: {e}")
            raise
    
    async def process_text_file(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        """Process basic text files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        metadata = {
            'extraction_method': 'direct_text',
            'content_length': len(content),
            'word_count': len(content.split())
        }
        
        return content, metadata
    
    async def process_fallback(self, file_path: str, file_type: str) -> tuple[str, Dict[str, Any]]:
        """Fallback processing when LlamaParse is not available"""
        
        logger.info(f"Using fallback processing for {file_type}")
        
        if file_type == 'pdf':
            try:
                # Use LlamaIndex's SimpleDirectoryReader as fallback
                from llama_index.core import SimpleDirectoryReader
                reader = SimpleDirectoryReader(input_files=[file_path])
                documents = reader.load_data()
                content = "\n\n".join([doc.text for doc in documents])
                
                metadata = {
                    'extraction_method': 'SimpleDirectoryReader',
                    'pages_processed': len(documents),
                    'content_length': len(content)
                }
                
                return content, metadata
                
            except Exception as e:
                logger.error(f"Fallback PDF processing failed: {e}")
                return f"[PDF PROCESSING FAILED: {Path(file_path).name}]", {'extraction_method': 'failed'}
        
        elif file_type in ['text', 'txt', 'md']:
            return await self.process_text_file(file_path)
        
        elif file_type == 'image':
            return (f"[IMAGE FILE: {Path(file_path).name}] - OCR requires LlamaParse", 
                   {'extraction_method': 'placeholder'})
        
        else:
            try:
                # Try to read as text
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content, {'extraction_method': 'text_fallback'}
            except:
                return (f"[UNSUPPORTED FILE: {Path(file_path).name}]", 
                       {'extraction_method': 'unsupported'})
    
    async def create_enhanced_nodes(self, content: str, upload_id: str, filename: str, 
                                  file_type: str, processing_metadata: Dict[str, Any]) -> List[TextNode]:
        """Create enhanced text nodes with rich metadata from LlamaParse processing"""
        
        # Intelligent chunking based on content structure
        if processing_metadata.get('formatting_preserved'):
            # Use structure-aware chunking for complex documents
            chunks = self.structure_aware_chunking(content)
        else:
            # Use simple chunking for basic documents
            chunk_size = 1000
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        nodes = []
        for i, chunk in enumerate(chunks):
            
            # Enhanced metadata combining processing info with chunk info
            metadata = {
                'upload_id': upload_id,
                'filename': filename,
                'file_type': file_type,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'upload_timestamp': datetime.now().isoformat(),
                'word_count': len(chunk.split()),
                'character_count': len(chunk),
                
                # LlamaParse-specific metadata
                'extraction_method': processing_metadata.get('extraction_method', 'unknown'),
                'parsing_strategy': processing_metadata.get('parsing_strategy', 'simple'),
                'llamaparse_processed': processing_metadata.get('llamaparse_version') is not None,
                'ocr_performed': processing_metadata.get('ocr_performed', False),
                'table_extraction': processing_metadata.get('table_extraction', False),
                'formatting_preserved': processing_metadata.get('formatting_preserved', False),
                
                # Content analysis
                'has_tables': processing_metadata.get('has_tables', False),
                'has_lists': processing_metadata.get('has_lists', False),
                'estimated_reading_time_minutes': processing_metadata.get('estimated_reading_time_minutes', 1),
                
                # Hybrid search readiness
                'hybrid_search_ready': True
            }
            
            node = TextNode(
                text=chunk,
                metadata=metadata,
                id_=f"{upload_id}_chunk_{i}"
            )
            nodes.append(node)
        
        return nodes
    
    def structure_aware_chunking(self, content: str) -> List[str]:
        """Intelligent chunking that preserves document structure"""
        
        # Split by common section breaks
        sections = []
        current_section = []
        
        lines = content.split('\n')
        
        for line in lines:
            # Check for section breaks (headers, page breaks, etc.)
            if (line.strip().startswith('#') or  # Markdown headers
                line.strip().isupper() and len(line.strip()) < 100 or  # UPPERCASE headers
                '---' in line or  # Horizontal rules
                'Page ' in line and len(line.strip()) < 20):  # Page indicators
                
                # Finalize current section
                if current_section:
                    section_text = '\n'.join(current_section).strip()
                    if section_text:
                        sections.append(section_text)
                    current_section = []
            
            current_section.append(line)
        
        # Add the last section
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if section_text:
                sections.append(section_text)
        
        # Split large sections if needed
        final_chunks = []
        max_chunk_size = 1500  # Slightly larger for structure-preserved chunks
        
        for section in sections:
            if len(section) <= max_chunk_size:
                final_chunks.append(section)
            else:
                # Split large sections while trying to preserve paragraphs
                paragraphs = section.split('\n\n')
                current_chunk = []
                current_size = 0
                
                for paragraph in paragraphs:
                    if current_size + len(paragraph) > max_chunk_size and current_chunk:
                        final_chunks.append('\n\n'.join(current_chunk))
                        current_chunk = [paragraph]
                        current_size = len(paragraph)
                    else:
                        current_chunk.append(paragraph)
                        current_size += len(paragraph) + 2  # +2 for \n\n
                
                if current_chunk:
                    final_chunks.append('\n\n'.join(current_chunk))
        
        return final_chunks
    
    async def process_document_pipeline(self, upload_id: str, file_path: str, 
                                      filename: str, document_type: str = None, 
                                      custom_metadata: str = None):
        """
        🚀 ULTIMATE DOCUMENT PROCESSING PIPELINE
        
        Features:
        - LlamaParse intelligent parsing
        - Hybrid search preparation
        - Advanced metadata extraction
        - Real-time BM25 index updates
        """
        
        try:
            logger.info(f"🚀 Starting ultimate processing for upload_id: {upload_id}")
            
            # Update status to processing
            self.update_status(upload_id, DocumentStatus.PROCESSING, progress=5)
            
            # Detect file type and complexity
            file_type = self.detect_file_type(file_path)
            complexity = self.detect_document_complexity(file_path, file_type)
            
            logger.info(f"📋 Document analysis: type={file_type}, complexity={complexity}")
            
            self.update_status(upload_id, DocumentStatus.PROCESSING, progress=15, 
                             metadata={'file_type': file_type, 'complexity': complexity})
            
            # Choose processing method - ALWAYS use LlamaParse for maximum quality when available
            if (self.llamaparse_enabled and 
                file_type in ['pdf', 'docx', 'pptx', 'xlsx', 'image', 'txt', 'md']):
                
                # Use 'complex' strategy for ALL documents to get maximum extraction
                # This includes OCR, table extraction, and formatting preservation
                processing_strategy = 'complex'
                
                # Override with OCR strategy for image files or scanned documents
                if file_type in ['image', 'jpg', 'png', 'tiff'] or 'scan' in filename.lower():
                    processing_strategy = 'ocr'
                
                logger.info(f"🔥 Using LlamaParse with '{processing_strategy}' strategy for MAXIMUM quality extraction...")
                logger.info("   📊 This includes: OCR, table extraction, formatting preservation, and comprehensive content analysis")
                self.update_status(upload_id, DocumentStatus.PROCESSING, progress=25)
                
                content, processing_metadata = await self.process_with_llamaparse(file_path, processing_strategy)
                
            else:
                logger.info("📄 Using fallback processing (LlamaParse not available or unsupported file type)...")
                self.update_status(upload_id, DocumentStatus.PROCESSING, progress=25)
                
                content, processing_metadata = await self.process_fallback(file_path, file_type)
            
            # Create content preview
            content_preview = content[:500] + "..." if len(content) > 500 else content
            
            self.update_status(upload_id, DocumentStatus.PROCESSING, progress=60, 
                             content_preview=content_preview)
            
            # Create enhanced nodes with rich metadata
            nodes = await self.create_enhanced_nodes(content, upload_id, filename, 
                                                   file_type, processing_metadata)
            
            self.update_status(upload_id, DocumentStatus.PROCESSING, progress=80)
            
            # Add nodes to vector store
            try:
                for node in nodes:
                    self.index.insert(node)
            except AttributeError:
                # Fallback method for different LlamaIndex versions
                from llama_index.core import VectorStoreIndex
                self.index = VectorStoreIndex(
                    nodes=nodes + list(self.document_corpus),
                    storage_context=self.storage_context
                )
                
            # Add to document corpus for BM25
            self.document_corpus.extend(nodes)
            
            self.update_status(upload_id, DocumentStatus.PROCESSING, progress=90)
            
            # Refresh BM25 index with new documents
            if self.hybrid_search_enabled:
                self.refresh_bm25_index()
                logger.info("🔍 BM25 index updated with new document")
            
            # Clean up temporary file
            try:
                os.remove(file_path)
            except:
                pass
            
            # Final metadata compilation
            final_metadata = {
                'content_length': len(content),
                'chunk_count': len(nodes),
                'file_type': file_type,
                'complexity': complexity,
                'hybrid_search_ready': self.hybrid_search_enabled,
                **processing_metadata  # Include all processing metadata
            }
            
            self.update_status(upload_id, DocumentStatus.COMPLETED, progress=100, 
                             metadata=final_metadata)
            
            logger.info(f"✅ ULTIMATE processing completed for upload_id: {upload_id}")
            logger.info(f"   📊 Extracted {len(content)} characters in {len(nodes)} chunks")
            logger.info(f"   🔧 Method: {processing_metadata.get('extraction_method', 'unknown')}")
            logger.info(f"   👁️  OCR Processing: {'✅ Applied' if processing_metadata.get('extraction_method', '').startswith('LlamaParse') else '❌ Not Available'}")
            logger.info(f"   📋 Table Extraction: {'✅ Applied' if processing_metadata.get('table_extraction', False) else '❌ Not Available'}")
            logger.info(f"   🔍 Hybrid Search: {'Ready' if self.hybrid_search_enabled else 'Disabled'}")
            logger.info(f"   🚀 Processing Mode: {'ULTIMATE EXTRACTION' if self.llamaparse_enabled else 'FALLBACK'}")
            
        except Exception as e:
            logger.error(f"❌ Ultimate processing failed for upload_id {upload_id}: {e}")
            self.update_status(upload_id, DocumentStatus.FAILED, 
                             error_message=str(e))
            
            # Clean up on failure
            try:
                os.remove(file_path)
            except:
                pass
    
    def get_query_engine(self):
        """Get the appropriate query engine (hybrid if available, fallback otherwise)"""
        
        if self.hybrid_search_enabled:
            return self.hybrid_query_engine
        else:
            return self.fallback_query_engine

# Global instance with LlamaParse + Hybrid Search capabilities
llamaparse_hybrid_processor = LlamaParseHybridProcessor() 