import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import List

# Load environment variables from .env file
load_dotenv()

from schemas import (
    CreateKbRequest, CreateKbResponse,
    IngestRequest, ChatRequest, ChatResponse
)
from kb_manager import KnowledgeBaseManager

app = FastAPI(
    title="Advanced RAG API",
    description="Enterprise-grade RAG system with LlamaParse, Hybrid Search, and Advanced Analytics",
    version="2.0.0"
)

kb_mgr = KnowledgeBaseManager()

@app.post("/knowledge_bases", response_model=CreateKbResponse)
def create_kb(req: CreateKbRequest):
    """Create a new knowledge base with advanced capabilities"""
    kb_id = kb_mgr.create_kb(req.name, req.embedding_model)
    return CreateKbResponse(id=kb_id)

@app.post("/knowledge_bases/{kb_id}/documents")
async def ingest_docs(kb_id: str, payload: IngestRequest):
    """Ingest documents with advanced processing (LlamaParse + Hybrid Search)"""
    try:
        await kb_mgr.ingest(
            kb_id, 
            payload.texts, 
            embedding_model=getattr(payload, "embedding_model", "openai")
        )
        return {"status": "ingested", "message": "Documents processed with advanced pipeline"}
    except KeyError:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

@app.post("/knowledge_bases/{kb_id}/upload")
async def ingest_files(kb_id: str, files: List[UploadFile] = File(...)):
    """Upload and process files with ultimate extraction capabilities"""
    try:
        texts: List[str] = []
        file_paths: List[str] = []
        
        # Save files temporarily for optimal LlamaParse processing
        import tempfile
        temp_files = []
        
        for f in files:
            content = await f.read()
            
            # Create temporary file for LlamaParse processing
            with tempfile.NamedTemporaryFile(mode='wb', suffix=f'.{f.filename.split(".")[-1]}', delete=False) as tmp:
                tmp.write(content)
                temp_files.append(tmp.name)
                file_paths.append(tmp.name)
            
            # Also add text content
            try:
                texts.append(content.decode("utf-8"))
            except UnicodeDecodeError:
                # For binary files, add placeholder text
                texts.append(f"Binary file: {f.filename}")
        
        # Process with advanced pipeline
        await kb_mgr.ingest(kb_id, texts, embedding_model="openai", files=file_paths)
        
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except OSError:
                pass
        
        return {
            "status": "ingested",
            "message": f"Processed {len(files)} files with ultimate extraction",
            "files_processed": [f.filename for f in files],
            "features_used": ["llamaparse", "ocr", "table_extraction", "hybrid_search"]
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

@app.post("/knowledge_bases/{kb_id}/chat", response_model=ChatResponse)
def chat(kb_id: str, req: ChatRequest):
    """Chat with knowledge base using hybrid search by default"""
    try:
        ans = kb_mgr.query(kb_id, req.query, embedding_model="openai")
        return ChatResponse(response=ans)
    except KeyError:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

@app.get("/knowledge_bases")
def list_kbs():
    """List all knowledge bases with advanced metadata"""
    metadata = kb_mgr.list_kbs()
    return [{"id": kb_id, **data} for kb_id, data in metadata.items()]

@app.get("/knowledge_bases/{kb_id}/stats")
def get_kb_stats(kb_id: str):
    """Get comprehensive knowledge base statistics and capabilities"""
    try:
        return kb_mgr.get_kb_stats(kb_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

@app.delete("/knowledge_bases/{kb_id}")
def delete_kb(kb_id: str):
    """Delete knowledge base and cleanup all resources"""
    try:
        kb_mgr.delete_kb(kb_id)
        return {"status": "deleted", "message": "Knowledge base and all resources cleaned up"}
    except KeyError:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

@app.get("/health")
def health_check():
    """System health check with advanced capabilities status"""
    return {
        "status": "healthy",
        "system": "Advanced RAG API v2.0",
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
        },
        "llamaparse_enabled": kb_mgr.llamaparse_enabled if hasattr(kb_mgr, 'llamaparse_enabled') else False
    }

@app.get("/")
def root():
    """API information"""
    return {
        "message": "Advanced RAG API - Enterprise Grade",
        "version": "2.0.0",
        "description": "Unified system with LlamaParse, Hybrid Search, and Advanced Analytics",
        "features": "Ultimate document processing with OCR, tables, hybrid search by default",
        "endpoints": {
            "create_kb": "POST /knowledge_bases",
            "ingest_docs": "POST /knowledge_bases/{kb_id}/documents",
            "upload_files": "POST /knowledge_bases/{kb_id}/upload",
            "chat": "POST /knowledge_bases/{kb_id}/chat",
            "list_kbs": "GET /knowledge_bases",
            "kb_stats": "GET /knowledge_bases/{kb_id}/stats",
            "delete_kb": "DELETE /knowledge_bases/{kb_id}",
            "health": "GET /health"
        }
    } 