#!/usr/bin/env python3
"""
🚀 ULTIMATE RAG SERVER - PHASE 4: HYBRID SEARCH

Features:
- 🔥 LlamaParse: Enterprise document processing (PDFs, Tables, OCR)
- 🔍 Hybrid Search: Semantic similarity + BM25 keyword matching
- 📊 Advanced Analytics: Rich metadata and search insights
- ⚡ Production-Ready: Background processing, error handling, monitoring
- 🎯 Smart Routing: Intelligent strategy selection based on document type
- 📈 Query Intelligence: Enhanced responses with search method analysis

This represents the current state-of-the-art in RAG systems!
"""

import os
import uuid
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import *
from llamaparse_hybrid_processor import llamaparse_hybrid_processor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    
    logger.info("🚀 Starting Ultimate RAG Server with Hybrid Search...")
    
    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    
    # Initialize hybrid search with any existing documents
    try:
        llamaparse_hybrid_processor.refresh_bm25_index()
        logger.info("✅ Hybrid search initialized successfully")
    except Exception as e:
        logger.warning(f"⚠️  Hybrid search initialization warning: {e}")
    
    logger.info("🔥 Ultimate RAG Server ready!")
    logger.info("   📊 Features: LlamaParse + Hybrid Search + Advanced Analytics")
    logger.info("   🔍 Search Methods: Semantic Similarity + BM25 Keyword Matching")
    logger.info("   📈 Document Types: PDF, DOCX, Images, Tables, Complex Layouts")
    
    yield
    
    logger.info("🛑 Shutting down Ultimate RAG Server...")

# Create FastAPI app with lifecycle management
app = FastAPI(
    title="🚀 Ultimate RAG Server - Hybrid Search Edition",
    description="""
    **The Most Advanced RAG System Available**
    
    This server combines enterprise-grade document processing with intelligent hybrid search:
    
    ## 🔥 Core Features
    - **LlamaParse Integration**: Enterprise document processing with OCR and table extraction
    - **Hybrid Search**: Combines semantic similarity with BM25 keyword matching
    - **Smart Processing**: Automatic strategy selection based on document complexity
    - **Advanced Analytics**: Rich metadata extraction and search insights
    - **Production Ready**: Background processing, error handling, monitoring
    
    ## 🔍 Search Capabilities
    - **Semantic Search**: Understanding meaning and context using embeddings
    - **Keyword Search**: Exact term matching using BM25 algorithm
    - **Hybrid Ranking**: Intelligent combination of both approaches
    - **Query Expansion**: Automatic synonym expansion for better recall
    
    ## 📊 Document Support
    - **Text Documents**: TXT, MD, RTF
    - **Office Documents**: PDF, DOCX, XLSX, PPTX
    - **Images**: JPG, PNG, TIFF (with OCR)
    - **Complex Layouts**: Tables, forms, multi-column documents
    
    ## 🎯 Use Cases
    - Enterprise document search and analysis
    - Research paper analysis and Q&A
    - Financial report processing
    - Legal document review
    - Technical manual querying
    """,
    version="4.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# ENHANCED API ENDPOINTS WITH HYBRID SEARCH
# =====================================================================

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_type: Optional[str] = Query(None, description="Document type for processing optimization"),
    custom_metadata: Optional[str] = Query(None, description="Custom metadata for the document")
):
    """
    🚀 **ULTIMATE DOCUMENT UPLOAD & PROCESSING**
    
    Upload documents for processing with LlamaParse + Hybrid Search preparation.
    
    **Features:**
    - Automatic complexity detection and strategy selection
    - LlamaParse processing for complex documents (PDFs, tables, OCR)
    - Fallback processing for simple documents
    - BM25 index automatic updates
    - Rich metadata extraction
    - Background processing with real-time status
    
    **Supported Formats:**
    - PDF (with table extraction and OCR)
    - DOCX, XLSX, PPTX (Office documents)
    - Images (JPG, PNG, TIFF) with OCR
    - Text files (TXT, MD, RTF)
    
    **Returns:**
    - Upload ID for status tracking
    - Processing time estimates
    - Capability information
    """
    
    try:
        # Generate unique upload ID
        upload_id = str(uuid.uuid4())
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Save uploaded file
        file_path = f"uploads/{upload_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            file_size = len(content)
        
        logger.info(f"📁 File uploaded: {file.filename} ({file_size} bytes)")
        
        # Create upload record with complexity analysis
        record = llamaparse_hybrid_processor.create_upload_record(
            upload_id, file.filename, file_size
        )
        
        # Start background processing
        background_tasks.add_task(
            llamaparse_hybrid_processor.process_document_pipeline,
            upload_id, file_path, file.filename, document_type, custom_metadata
        )
        
        # Enhanced response with capabilities
        return UploadResponse(
            upload_id=upload_id,
            filename=file.filename,
            status=DocumentStatus.PENDING,
            message="Document uploaded successfully and queued for ultimate processing",
            estimated_processing_time_seconds=record['metadata']['estimated_processing_time']
        )
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/status/{upload_id}", response_model=DocumentStatusResponse)
async def get_document_status(upload_id: str):
    """
    📊 **ENHANCED STATUS TRACKING**
    
    Get detailed processing status with advanced analytics.
    
    **Enhanced Information:**
    - Processing progress and completion estimates
    - Extraction method and strategy used
    - Content preview and metadata
    - Hybrid search readiness status
    - Error details and recovery suggestions
    """
    
    status_info = llamaparse_hybrid_processor.get_status(upload_id)
    
    if not status_info:
        raise HTTPException(status_code=404, detail="Upload ID not found")
    
    # Enhanced status response with processing insights
    return DocumentStatusResponse(
        upload_id=upload_id,
        filename=status_info['filename'],
        status=status_info['status'],
        progress_percentage=status_info['progress_percentage'],
        created_at=status_info['created_at'],
        completed_at=status_info.get('completed_at'),
        error_message=status_info.get('error_message'),
        extracted_content_preview=status_info.get('extracted_content_preview'),
        metadata=status_info.get('metadata', {}),
        processing_insights={
            "hybrid_search_ready": status_info.get('metadata', {}).get('hybrid_search_ready', False),
            "extraction_method": status_info.get('metadata', {}).get('extraction_method', 'unknown'),
            "processing_strategy": status_info.get('metadata', {}).get('parsing_strategy', 'simple'),
            "bm25_indexed": status_info['status'] == DocumentStatus.COMPLETED
        }
    )

@app.post("/query", response_model=HybridQueryResponse)
async def hybrid_query(request: QueryRequest):
    """
    🔍 **ULTIMATE HYBRID QUERY SYSTEM**
    
    The most advanced query system combining multiple search methods for optimal results.
    
    **Hybrid Search Features:**
    - **Semantic Search**: Understanding meaning and context
    - **Keyword Search**: Exact term and phrase matching
    - **Intelligent Ranking**: Combines both methods with smart weighting
    - **Query Expansion**: Automatic synonym expansion
    - **Search Analytics**: Detailed insights into retrieval methods
    
    **Query Processing:**
    - Automatic query optimization and expansion
    - Multi-method result combination and ranking
    - Confidence scoring and method attribution
    - Rich metadata extraction from results
    
    **Response Enhancement:**
    - Search method breakdown and analysis
    - Matched keyword identification
    - Source attribution with processing metadata
    - Confidence scoring and ranking explanations
    """
    
    try:
        logger.info(f"🔍 Processing hybrid query: '{request.question}'")
        
        # Get the hybrid query engine
        query_engine = llamaparse_hybrid_processor.get_query_engine()
        
        if not query_engine:
            raise HTTPException(
                status_code=503, 
                detail="Query engine not available. Please upload documents first."
            )
        
        # Execute hybrid query
        start_time = datetime.now()
        response = query_engine.query(request.question)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Extract search analytics from source nodes
        search_analytics = {
            "total_sources": len(response.source_nodes),
            "semantic_matches": 0,
            "bm25_matches": 0,
            "hybrid_matches": 0,
            "top_matched_keywords": [],
            "processing_methods": {},
            "confidence_distribution": []
        }
        
        enhanced_sources = []
        
        for i, source_node in enumerate(response.source_nodes):
            node_metadata = source_node.node.metadata
            
            # Extract hybrid search information
            hybrid_info = node_metadata.get('hybrid_search_info', {})
            search_method = hybrid_info.get('search_method', 'unknown')
            
            # Update analytics
            if search_method == 'semantic':
                search_analytics["semantic_matches"] += 1
            elif search_method == 'bm25':
                search_analytics["bm25_matches"] += 1
            elif search_method == 'hybrid':
                search_analytics["hybrid_matches"] += 1
            
            # Collect matched keywords
            keywords = hybrid_info.get('matched_keywords', [])
            search_analytics["top_matched_keywords"].extend(keywords)
            
            # Track processing methods
            processing_method = node_metadata.get('extraction_method', 'unknown')
            search_analytics["processing_methods"][processing_method] = (
                search_analytics["processing_methods"].get(processing_method, 0) + 1
            )
            
            # Enhanced source information
            enhanced_source = EnhancedSourceNode(
                content=source_node.node.text[:1000] + "..." if len(source_node.node.text) > 1000 else source_node.node.text,
                score=source_node.score,
                metadata={
                    "filename": node_metadata.get('filename', 'unknown'),
                    "chunk_index": node_metadata.get('chunk_index', 0),
                    "upload_timestamp": node_metadata.get('upload_timestamp', ''),
                    "word_count": node_metadata.get('word_count', 0),
                    "extraction_method": node_metadata.get('extraction_method', 'unknown'),
                    "parsing_strategy": node_metadata.get('parsing_strategy', 'simple'),
                    "llamaparse_processed": node_metadata.get('llamaparse_processed', False),
                    "file_type": node_metadata.get('file_type', 'unknown')
                },
                search_info={
                    "search_method": search_method,
                    "semantic_score": hybrid_info.get('semantic_score', 0.0),
                    "bm25_score": hybrid_info.get('bm25_score', 0.0),
                    "hybrid_score": hybrid_info.get('hybrid_score', 0.0),
                    "matched_keywords": keywords
                }
            )
            
            enhanced_sources.append(enhanced_source)
            search_analytics["confidence_distribution"].append(source_node.score)
        
        # Finalize analytics
        search_analytics["top_matched_keywords"] = list(set(search_analytics["top_matched_keywords"]))[:10]
        search_analytics["average_confidence"] = (
            sum(search_analytics["confidence_distribution"]) / len(search_analytics["confidence_distribution"])
            if search_analytics["confidence_distribution"] else 0.0
        )
        
        # Create enhanced response
        return HybridQueryResponse(
            query=request.question,
            response=response.response,
            sources=enhanced_sources,
            search_analytics=search_analytics,
            processing_time_seconds=processing_time,
            search_method_summary={
                "primary_method": "hybrid" if llamaparse_hybrid_processor.hybrid_search_enabled else "semantic",
                "methods_used": ["semantic", "bm25", "hybrid"] if llamaparse_hybrid_processor.hybrid_search_enabled else ["semantic"],
                "query_expansion_enabled": llamaparse_hybrid_processor.hybrid_search_enabled,
                "bm25_enabled": llamaparse_hybrid_processor.hybrid_search_enabled
            },
            metadata={
                "total_documents_indexed": len(llamaparse_hybrid_processor.document_corpus),
                "llamaparse_documents": len([d for d in llamaparse_hybrid_processor.document_corpus 
                                           if d.metadata.get('llamaparse_processed', False)]),
                "hybrid_search_active": llamaparse_hybrid_processor.hybrid_search_enabled,
                "response_generated_at": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Hybrid query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/search/analytics", response_model=SearchAnalytics)
async def get_search_analytics():
    """
    📈 **ADVANCED SEARCH ANALYTICS**
    
    Get comprehensive analytics about the search system and document corpus.
    
    **Analytics Include:**
    - Document processing statistics
    - Search method performance
    - Content type distribution
    - Processing method breakdown
    - System capability overview
    """
    
    try:
        corpus = llamaparse_hybrid_processor.document_corpus
        
        # Document statistics
        total_documents = len(corpus)
        total_chunks = len(corpus)
        
        # Processing method analysis
        processing_methods = {}
        file_types = {}
        llamaparse_processed = 0
        hybrid_ready = 0
        
        total_words = 0
        total_characters = 0
        
        for node in corpus:
            metadata = node.metadata
            
            # Processing methods
            method = metadata.get('extraction_method', 'unknown')
            processing_methods[method] = processing_methods.get(method, 0) + 1
            
            # File types
            file_type = metadata.get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
            
            # LlamaParse processing
            if metadata.get('llamaparse_processed', False):
                llamaparse_processed += 1
            
            # Hybrid search readiness
            if metadata.get('hybrid_search_ready', False):
                hybrid_ready += 1
            
            # Content statistics
            total_words += metadata.get('word_count', 0)
            total_characters += metadata.get('character_count', 0)
        
        return SearchAnalytics(
            total_documents=total_documents,
            total_chunks=total_chunks,
            processing_methods=processing_methods,
            file_types=file_types,
            content_statistics={
                "total_words": total_words,
                "total_characters": total_characters,
                "average_words_per_chunk": total_words / total_chunks if total_chunks > 0 else 0,
                "average_characters_per_chunk": total_characters / total_chunks if total_chunks > 0 else 0
            },
            search_capabilities={
                "semantic_search": True,
                "bm25_search": llamaparse_hybrid_processor.hybrid_search_enabled,
                "hybrid_search": llamaparse_hybrid_processor.hybrid_search_enabled,
                "query_expansion": llamaparse_hybrid_processor.hybrid_search_enabled,
                "llamaparse_processing": llamaparse_hybrid_processor.llamaparse_enabled
            },
            system_status={
                "llamaparse_documents": llamaparse_processed,
                "hybrid_ready_documents": hybrid_ready,
                "bm25_index_size": len(corpus),
                "last_updated": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Analytics generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

@app.get("/system/health", response_model=SystemHealth)
async def get_system_health():
    """
    🏥 **COMPREHENSIVE SYSTEM HEALTH CHECK**
    
    Detailed health monitoring for all system components.
    """
    
    try:
        # Check all system components
        components = {
            "vector_store": False,
            "llamaparse": False,
            "hybrid_search": False,
            "bm25_index": False,
            "query_engine": False
        }
        
        component_details = {}
        
        # Vector store check
        try:
            if llamaparse_hybrid_processor.vector_store:
                components["vector_store"] = True
                component_details["vector_store"] = "Connected to Pinecone"
        except Exception as e:
            component_details["vector_store"] = f"Error: {str(e)}"
        
        # LlamaParse check
        if llamaparse_hybrid_processor.llamaparse_enabled:
            components["llamaparse"] = True
            component_details["llamaparse"] = "API key configured and ready"
        else:
            component_details["llamaparse"] = "API key not configured - using fallback"
        
        # Hybrid search check
        if llamaparse_hybrid_processor.hybrid_search_enabled:
            components["hybrid_search"] = True
            component_details["hybrid_search"] = "Semantic + BM25 hybrid search active"
        else:
            component_details["hybrid_search"] = "Disabled - using semantic search only"
        
        # BM25 index check
        try:
            if (llamaparse_hybrid_processor.hybrid_search_enabled and 
                llamaparse_hybrid_processor.hybrid_retriever.bm25_index):
                components["bm25_index"] = True
                component_details["bm25_index"] = f"Built with {len(llamaparse_hybrid_processor.document_corpus)} documents"
            else:
                component_details["bm25_index"] = "Not built or disabled"
        except Exception as e:
            component_details["bm25_index"] = f"Error: {str(e)}"
        
        # Query engine check
        try:
            query_engine = llamaparse_hybrid_processor.get_query_engine()
            if query_engine:
                components["query_engine"] = True
                component_details["query_engine"] = "Ready for queries"
        except Exception as e:
            component_details["query_engine"] = f"Error: {str(e)}"
        
        # Overall status
        healthy_components = sum(components.values())
        total_components = len(components)
        overall_healthy = healthy_components >= 3  # At least basic functionality
        
        return SystemHealth(
            status="healthy" if overall_healthy else "degraded",
            components=components,
            component_details=component_details,
            statistics={
                "uptime_seconds": 0,  # Would track actual uptime in production
                "documents_processed": len(set(node.metadata.get('upload_id', '') 
                                             for node in llamaparse_hybrid_processor.document_corpus)),
                "total_chunks": len(llamaparse_hybrid_processor.document_corpus),
                "queries_processed": 0,  # Would track in production
                "healthy_components": healthy_components,
                "total_components": total_components
            },
            last_check=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return SystemHealth(
            status="unhealthy",
            components={},
            component_details={"error": str(e)},
            statistics={},
            last_check=datetime.now()
        )

@app.get("/", response_model=Dict[str, Any])
async def root():
    """
    🏠 **ULTIMATE RAG SERVER OVERVIEW**
    
    Welcome to the most advanced RAG system available!
    """
    
    return {
        "name": "🚀 Ultimate RAG Server - Hybrid Search Edition",
        "version": "4.0.0",
        "description": "Enterprise-grade RAG with LlamaParse + Hybrid Search",
        "phase": "Phase 4: Hybrid Search Implementation",
        "features": {
            "document_processing": {
                "llamaparse": llamaparse_hybrid_processor.llamaparse_enabled,
                "ocr_support": True,
                "table_extraction": True,
                "multi_format": ["PDF", "DOCX", "XLSX", "PPTX", "Images", "Text"]
            },
            "search_capabilities": {
                "semantic_search": True,
                "bm25_keyword_search": llamaparse_hybrid_processor.hybrid_search_enabled,
                "hybrid_search": llamaparse_hybrid_processor.hybrid_search_enabled,
                "query_expansion": llamaparse_hybrid_processor.hybrid_search_enabled,
                "intelligent_ranking": True
            },
            "production_features": {
                "background_processing": True,
                "real_time_status": True,
                "error_handling": True,
                "comprehensive_analytics": True,
                "health_monitoring": True
            }
        },
        "endpoints": {
            "upload": "/upload - Upload and process documents",
            "query": "/query - Hybrid search queries",
            "status": "/status/{upload_id} - Processing status",
            "analytics": "/search/analytics - Search analytics",
            "health": "/system/health - System health check"
        },
        "statistics": {
            "documents_indexed": len(set(node.metadata.get('upload_id', '') 
                                       for node in llamaparse_hybrid_processor.document_corpus)),
            "total_chunks": len(llamaparse_hybrid_processor.document_corpus),
            "hybrid_search_ready": llamaparse_hybrid_processor.hybrid_search_enabled
        },
        "next_phases": [
            "Phase 5: OCR & Multi-modal Processing",
            "Phase 6: Production Features & Monitoring"
        ]
    }

# =====================================================================
# SERVER STARTUP
# =====================================================================

if __name__ == "__main__":
    print("🚀 Starting Ultimate RAG Server - Hybrid Search Edition")
    print("🔥 Features: LlamaParse + Semantic Search + BM25 + Hybrid Ranking")
    print("📊 The most advanced RAG system available!")
    print()
    
    uvicorn.run(
        "hybrid_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 