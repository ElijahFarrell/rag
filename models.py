from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PENDING = "pending"

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask the knowledge base")
    max_results: Optional[int] = Field(5, description="Maximum number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters to apply")
    include_sources: Optional[bool] = Field(True, description="Whether to include source documents")

class Source(BaseModel):
    content: str = Field(..., description="Excerpt from the source document")
    score: Optional[float] = Field(None, description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    document_id: Optional[str] = Field(None, description="Unique document identifier")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The generated answer")
    confidence: float = Field(..., description="Confidence score between 0 and 1")
    sources: List[Source] = Field(default_factory=list, description="Source documents used")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    query_id: str = Field(..., description="Unique query identifier")

class UploadResponse(BaseModel):
    upload_id: str = Field(..., description="Unique upload identifier")
    filename: str = Field(..., description="Original filename")
    status: DocumentStatus = Field(..., description="Current processing status")
    message: str = Field(..., description="Status message")
    estimated_processing_time_seconds: int = Field(..., description="Estimated processing time")

class DocumentStatusResponse(BaseModel):
    upload_id: str = Field(..., description="Unique upload identifier")
    filename: str = Field(..., description="Original filename")
    status: DocumentStatus = Field(..., description="Current processing status")
    progress_percentage: int = Field(..., description="Processing progress (0-100)")
    extracted_content_preview: Optional[str] = Field(None, description="Preview of extracted content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Extracted document metadata")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    created_at: datetime = Field(..., description="Upload timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    processing_insights: Optional[Dict[str, Any]] = Field(None, description="Processing method insights")

class KnowledgeBaseStats(BaseModel):
    total_documents: int = Field(..., description="Total number of documents")
    total_chunks: int = Field(..., description="Total number of text chunks")
    supported_file_types: List[str] = Field(..., description="Supported file formats")
    features: List[str] = Field(..., description="Available processing features")
    index_size_mb: Optional[float] = Field(None, description="Vector index size in MB")

class SearchSuggestion(BaseModel):
    suggestion: str = Field(..., description="Suggested query")
    confidence: float = Field(..., description="Confidence in suggestion relevance")

class SearchSuggestionsResponse(BaseModel):
    suggestions: List[SearchSuggestion] = Field(..., description="Query suggestions")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall system status")
    rag_system: str = Field(..., description="RAG system status")
    embedding_model: str = Field(..., description="Embedding model being used")
    vector_store: str = Field(..., description="Vector store status")
    background_tasks: int = Field(..., description="Number of pending background tasks")
    timestamp: datetime = Field(..., description="Health check timestamp")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")

# Enhanced models for hybrid search
class EnhancedSourceNode(BaseModel):
    """Enhanced source node with hybrid search information"""
    content: str
    score: float
    metadata: Dict[str, Any]
    search_info: Dict[str, Any]  # Hybrid search specific info

class HybridQueryResponse(BaseModel):
    """Enhanced response with hybrid search analytics"""
    query: str
    response: str
    sources: List[EnhancedSourceNode]
    search_analytics: Dict[str, Any]
    processing_time_seconds: float
    search_method_summary: Dict[str, Any]
    metadata: Dict[str, Any]

class SearchAnalytics(BaseModel):
    """Comprehensive search analytics"""
    total_documents: int
    total_chunks: int
    processing_methods: Dict[str, int]
    file_types: Dict[str, int]
    content_statistics: Dict[str, Any]
    search_capabilities: Dict[str, bool]
    system_status: Dict[str, Any]

class SystemHealth(BaseModel):
    """System health monitoring"""
    status: str
    components: Dict[str, bool]
    component_details: Dict[str, str]
    statistics: Dict[str, Any]
    last_check: datetime 