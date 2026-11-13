import os
import asyncio
from dotenv import load_dotenv
from pathlib import Path
import tempfile

from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from typing import List

from api_security import read_limited_upload, require_api_key, safe_upload_filename

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

@app.post(
    "/knowledge_bases",
    response_model=CreateKbResponse,
    dependencies=[Depends(require_api_key)],
)
def create_kb(req: CreateKbRequest):
    """Create a new knowledge base with advanced capabilities"""
    kb_id = kb_mgr.create_kb(req.name, req.embedding_model)
    return CreateKbResponse(id=kb_id)

@app.post(
    "/knowledge_bases/{kb_id}/documents",
    dependencies=[Depends(require_api_key)],
)
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

@app.post(
    "/knowledge_bases/{kb_id}/upload",
    dependencies=[Depends(require_api_key)],
)
async def ingest_files(kb_id: str, files: List[UploadFile] = File(...)):
    """Upload and process files with ultimate extraction capabilities"""
    texts: List[str] = []
    file_paths: List[str] = []
    processed_names: List[str] = []
    temp_files: List[str] = []

    try:
        for f in files:
            safe_name = safe_upload_filename(f.filename)
            content = await read_limited_upload(f)
            processed_names.append(safe_name)

            # Extension comes from the sanitized name, never from a path in the upload.
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=Path(safe_name).suffix, delete=False
            ) as tmp:
                tmp.write(content)
                temp_files.append(tmp.name)
                file_paths.append(tmp.name)

            try:
                texts.append(content.decode("utf-8"))
            except UnicodeDecodeError:
                texts.append(f"Binary file: {safe_name}")

        await kb_mgr.ingest(kb_id, texts, embedding_model="openai", files=file_paths)
        return {
            "status": "ingested",
            "message": f"Processed {len(files)} files with ultimate extraction",
            "files_processed": processed_names,
            "features_used": ["llamaparse", "ocr", "table_extraction", "hybrid_search"]
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    finally:
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except OSError:
                pass

@app.post(
    "/knowledge_bases/{kb_id}/chat",
    response_model=ChatResponse,
    dependencies=[Depends(require_api_key)],
)
def chat(kb_id: str, req: ChatRequest):
    """Chat with knowledge base using hybrid search by default"""
    try:
        ans = kb_mgr.query(kb_id, req.query, embedding_model="openai")
        return ChatResponse(response=ans)
    except KeyError:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

@app.get("/knowledge_bases", dependencies=[Depends(require_api_key)])
def list_kbs():
    """List all knowledge bases with advanced metadata"""
    metadata = kb_mgr.list_kbs()
    return [{"id": kb_id, **data} for kb_id, data in metadata.items()]

@app.get("/knowledge_bases/{kb_id}/stats", dependencies=[Depends(require_api_key)])
def get_kb_stats(kb_id: str):
    """Get comprehensive knowledge base statistics and capabilities"""
    try:
        return kb_mgr.get_kb_stats(kb_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

@app.delete("/knowledge_bases/{kb_id}", dependencies=[Depends(require_api_key)])
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
        "auth": "Authorization: Bearer <RAG_API_KEY> is required for every /knowledge_bases route",
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