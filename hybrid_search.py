import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from rank_bm25 import BM25Okapi

from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores import VectorStoreQuery, VectorStoreQueryResult
from llama_index.core.embeddings import BaseEmbedding

logger = logging.getLogger(__name__)

@dataclass
class HybridSearchResult:
    """Result from hybrid search combining semantic and keyword scores"""
    node: TextNode
    semantic_score: float
    bm25_score: float
    hybrid_score: float
    search_method: str  # 'semantic', 'bm25', or 'hybrid'
    matched_keywords: List[str]

class HybridRetriever(BaseRetriever):
    """
    🔍 HYBRID SEARCH RETRIEVER
    
    Combines semantic similarity search with BM25 keyword matching for:
    - Better recall (finds more relevant documents)
    - Improved precision (better ranking of results)
    - Handling of exact term matches
    - Support for both conceptual and literal queries
    """
    
    def __init__(
        self,
        vector_store,
        embed_model: BaseEmbedding,
        similarity_top_k: int = 10,
        bm25_top_k: int = 10,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        hybrid_threshold: float = 0.5,
        enable_query_expansion: bool = True
    ):
        """
        Initialize the hybrid retriever
        
        Args:
            vector_store: Vector store for semantic search
            embed_model: Embedding model for query encoding
            similarity_top_k: Number of results from semantic search
            bm25_top_k: Number of results from BM25 search
            semantic_weight: Weight for semantic scores (0-1)
            bm25_weight: Weight for BM25 scores (0-1)
            hybrid_threshold: Minimum score threshold for hybrid results
            enable_query_expansion: Whether to expand queries with synonyms
        """
        super().__init__()
        
        self.vector_store = vector_store
        self.embed_model = embed_model
        self.similarity_top_k = similarity_top_k
        self.bm25_top_k = bm25_top_k
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.hybrid_threshold = hybrid_threshold
        self.enable_query_expansion = enable_query_expansion
        
        # BM25 components
        self.bm25_index = None
        self.text_corpus = []
        self.node_map = {}  # Maps corpus index to node
        
        # Query expansion terms
        self.query_expansions = {
            'financial': ['revenue', 'profit', 'earnings', 'income', 'sales'],
            'performance': ['results', 'metrics', 'outcomes', 'achievement'],
            'increase': ['growth', 'rise', 'improvement', 'gain', 'boost'],
            'decrease': ['decline', 'drop', 'reduction', 'fall', 'loss'],
            'table': ['chart', 'data', 'statistics', 'figures', 'numbers'],
            'report': ['document', 'summary', 'analysis', 'findings']
        }
        
        logger.info("🔍 Hybrid Retriever initialized")
    
    def build_bm25_index(self, nodes: List[TextNode]):
        """Build BM25 index from document nodes"""
        
        logger.info(f"🔍 Building BM25 index from {len(nodes)} nodes...")
        
        # Prepare text corpus for BM25
        self.text_corpus = []
        self.node_map = {}
        
        for i, node in enumerate(nodes):
            # Preprocess text for better BM25 performance
            processed_text = self.preprocess_text_for_bm25(node.text)
            tokenized_text = processed_text.split()
            
            self.text_corpus.append(tokenized_text)
            self.node_map[i] = node
        
        # Build BM25 index
        if self.text_corpus:
            self.bm25_index = BM25Okapi(self.text_corpus)
            logger.info(f"✅ BM25 index built with {len(self.text_corpus)} documents")
        else:
            logger.warning("⚠️  No documents available for BM25 indexing")
    
    def preprocess_text_for_bm25(self, text: str) -> str:
        """Preprocess text for optimal BM25 performance"""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\-\.\%\$]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Extract numbers and percentages as separate tokens
        # This helps with financial data matching
        text = re.sub(r'(\d+\.?\d*%?)', r' \1 ', text)
        
        return text
    
    def expand_query(self, query: str) -> str:
        """Expand query with relevant synonyms for better recall"""
        
        if not self.enable_query_expansion:
            return query
        
        expanded_terms = []
        query_lower = query.lower()
        
        for key, synonyms in self.query_expansions.items():
            if key in query_lower:
                expanded_terms.extend(synonyms)
        
        if expanded_terms:
            # Add expanded terms with lower weight (indicated by parentheses)
            expansion = ' (' + ' '.join(expanded_terms) + ')'
            return query + expansion
        
        return query
    
    def semantic_search(self, query: str) -> List[NodeWithScore]:
        """Perform semantic similarity search"""
        
        try:
            # Create vector store query
            query_embedding = self.embed_model.get_text_embedding(query)
            
            vector_store_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=self.similarity_top_k,
                mode="default"
            )
            
            # Execute semantic search
            query_result = self.vector_store.query(vector_store_query)
            
            # Convert to NodeWithScore objects
            semantic_results = []
            if query_result.nodes:
                for i, node in enumerate(query_result.nodes):
                    score = query_result.similarities[i] if query_result.similarities else 0.0
                    semantic_results.append(NodeWithScore(node=node, score=score))
            
            logger.info(f"🧠 Semantic search returned {len(semantic_results)} results")
            return semantic_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def bm25_search(self, query: str) -> List[Tuple[TextNode, float]]:
        """Perform BM25 keyword search"""
        
        if not self.bm25_index:
            logger.warning("⚠️  BM25 index not available")
            return []
        
        try:
            # Expand and preprocess query
            expanded_query = self.expand_query(query)
            processed_query = self.preprocess_text_for_bm25(expanded_query)
            tokenized_query = processed_query.split()
            
            # Get BM25 scores
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top results
            top_indices = np.argsort(bm25_scores)[::-1][:self.bm25_top_k]
            
            bm25_results = []
            for idx in top_indices:
                if bm25_scores[idx] > 0:  # Only include non-zero scores
                    node = self.node_map[idx]
                    score = float(bm25_scores[idx])
                    bm25_results.append((node, score))
            
            logger.info(f"🔍 BM25 search returned {len(bm25_results)} results")
            return bm25_results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def extract_matched_keywords(self, query: str, text: str) -> List[str]:
        """Extract keywords from query that appear in the text"""
        
        query_terms = set(self.preprocess_text_for_bm25(query).split())
        text_terms = set(self.preprocess_text_for_bm25(text).split())
        
        matched = list(query_terms.intersection(text_terms))
        return matched[:5]  # Limit to top 5 matches
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range using min-max normalization"""
        
        if not scores or len(scores) == 1:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        normalized = [(score - min_score) / (max_score - min_score) for score in scores]
        return normalized
    
    def combine_results(self, query: str, semantic_results: List[NodeWithScore], 
                       bm25_results: List[Tuple[TextNode, float]]) -> List[HybridSearchResult]:
        """Combine and rank results from both search methods"""
        
        # Create lookup maps
        semantic_map = {node.node.id_: (node.node, node.score) for node in semantic_results}
        bm25_map = {node.id_: (node, score) for node, score in bm25_results}
        
        # Get all unique nodes
        all_node_ids = set(semantic_map.keys()) | set(bm25_map.keys())
        
        # Normalize scores separately
        semantic_scores = [result.score for result in semantic_results]
        bm25_scores = [score for _, score in bm25_results]
        
        normalized_semantic = self.normalize_scores(semantic_scores)
        normalized_bm25 = self.normalize_scores(bm25_scores)
        
        # Update maps with normalized scores
        if normalized_semantic:
            for i, result in enumerate(semantic_results):
                semantic_map[result.node.id_] = (result.node, normalized_semantic[i])
        
        if normalized_bm25:
            for i, (node, _) in enumerate(bm25_results):
                bm25_map[node.id_] = (node, normalized_bm25[i])
        
        # Combine results
        hybrid_results = []
        
        for node_id in all_node_ids:
            semantic_data = semantic_map.get(node_id, (None, 0.0))
            bm25_data = bm25_map.get(node_id, (None, 0.0))
            
            # Get the node (prefer semantic result if available)
            node = semantic_data[0] if semantic_data[0] else bm25_data[0]
            if not node:
                continue
                
            semantic_score = semantic_data[1]
            bm25_score = bm25_data[1]
            
            # Calculate hybrid score
            hybrid_score = (self.semantic_weight * semantic_score + 
                          self.bm25_weight * bm25_score)
            
            # Determine search method
            if semantic_score > 0 and bm25_score > 0:
                search_method = 'hybrid'
            elif semantic_score > 0:
                search_method = 'semantic'
            else:
                search_method = 'bm25'
            
            # Extract matched keywords
            matched_keywords = self.extract_matched_keywords(query, node.text)
            
            # Only include results above threshold
            if hybrid_score >= self.hybrid_threshold:
                hybrid_results.append(HybridSearchResult(
                    node=node,
                    semantic_score=semantic_score,
                    bm25_score=bm25_score,
                    hybrid_score=hybrid_score,
                    search_method=search_method,
                    matched_keywords=matched_keywords
                ))
        
        # Sort by hybrid score (descending)
        hybrid_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        logger.info(f"🔍 Combined search: {len(hybrid_results)} hybrid results")
        logger.info(f"   🧠 Semantic: {len([r for r in hybrid_results if r.search_method in ['semantic', 'hybrid']])}")
        logger.info(f"   🔍 BM25: {len([r for r in hybrid_results if r.search_method in ['bm25', 'hybrid']])}")
        logger.info(f"   ⚡ Hybrid: {len([r for r in hybrid_results if r.search_method == 'hybrid'])}")
        
        return hybrid_results
    
    def _retrieve(self, query_bundle) -> List[NodeWithScore]:
        """Main retrieval method called by LlamaIndex"""
        
        query = query_bundle.query_str
        logger.info(f"🔍 Executing hybrid search for: '{query}'")
        
        # Perform both types of search
        semantic_results = self.semantic_search(query)
        bm25_results = self.bm25_search(query)
        
        # Combine results
        hybrid_results = self.combine_results(query, semantic_results, bm25_results)
        
        # Convert back to NodeWithScore format
        final_results = []
        for result in hybrid_results:
            node_with_score = NodeWithScore(
                node=result.node,
                score=result.hybrid_score
            )
            
            # Add hybrid search metadata
            if not hasattr(result.node, 'metadata'):
                result.node.metadata = {}
                
            result.node.metadata.update({
                'hybrid_search_info': {
                    'semantic_score': result.semantic_score,
                    'bm25_score': result.bm25_score,
                    'hybrid_score': result.hybrid_score,
                    'search_method': result.search_method,
                    'matched_keywords': result.matched_keywords
                }
            })
            
            final_results.append(node_with_score)
        
        logger.info(f"✅ Hybrid search completed: {len(final_results)} final results")
        
        return final_results

class HybridQueryEngine:
    """
    🚀 HYBRID QUERY ENGINE
    
    High-level interface for hybrid search with enhanced query processing
    """
    
    def __init__(self, hybrid_retriever: HybridRetriever, llm, response_synthesizer=None):
        self.hybrid_retriever = hybrid_retriever
        self.llm = llm
        self.response_synthesizer = response_synthesizer
        
    def query(self, query_str: str, **kwargs):
        """Execute hybrid query with enhanced processing"""
        
        from llama_index.core import QueryBundle
        
        logger.info(f"🚀 Processing hybrid query: '{query_str}'")
        
        # Create query bundle
        query_bundle = QueryBundle(query_str=query_str)
        
        # Retrieve using hybrid search
        retrieved_nodes = self.hybrid_retriever.retrieve(query_bundle)
        
        if not retrieved_nodes:
            # Create a simple response object
            class SimpleResponse:
                def __init__(self, response_text, source_nodes):
                    self.response = response_text
                    self.source_nodes = source_nodes
            
            return SimpleResponse(
                "I couldn't find any relevant information for your query.",
                []
            )
        
        # Analyze search results for query insights
        search_analysis = self.analyze_search_results(retrieved_nodes)
        
        # Enhanced context preparation with search method info
        context_parts = []
        for i, node_with_score in enumerate(retrieved_nodes[:5]):  # Top 5 results
            node = node_with_score.node
            search_info = node.metadata.get('hybrid_search_info', {})
            
            # Create rich context with search metadata
            context_parts.append(f"""
Source {i+1} (Score: {node_with_score.score:.3f}, Method: {search_info.get('search_method', 'unknown')}):
{node.text}

Keywords: {', '.join(search_info.get('matched_keywords', []))}
---""")
        
        context = "\n".join(context_parts)
        
        # Enhanced prompt with search method awareness
        enhanced_prompt = f"""
Based on the following search results from both semantic similarity and keyword matching:

{context}

Search Analysis:
- Total results found: {len(retrieved_nodes)}
- Semantic matches: {search_analysis['semantic_count']}
- Keyword matches: {search_analysis['bm25_count']}
- Hybrid matches: {search_analysis['hybrid_count']}
- Best matched keywords: {', '.join(search_analysis['top_keywords'])}

Question: {query_str}

Please provide a comprehensive answer based on the search results. Consider both the semantic meaning and exact keyword matches in your response.
"""
        
        # Generate response using LLM
        if self.response_synthesizer:
            response = self.response_synthesizer.synthesize(query_bundle, retrieved_nodes)
        else:
            # Simple LLM call
            response_text = self.llm.complete(enhanced_prompt).text
            
            # Create simple response object
            class SimpleResponse:
                def __init__(self, response_text, source_nodes):
                    self.response = response_text
                    self.source_nodes = source_nodes
            
            response = SimpleResponse(
                response_text,
                retrieved_nodes
            )
        
        logger.info(f"✅ Hybrid query completed with {len(retrieved_nodes)} source nodes")
        
        return response
    
    def analyze_search_results(self, retrieved_nodes: List[NodeWithScore]) -> Dict[str, Any]:
        """Analyze search results to provide insights"""
        
        semantic_count = 0
        bm25_count = 0
        hybrid_count = 0
        all_keywords = []
        
        for node_with_score in retrieved_nodes:
            search_info = node_with_score.node.metadata.get('hybrid_search_info', {})
            method = search_info.get('search_method', 'unknown')
            
            if method == 'semantic':
                semantic_count += 1
            elif method == 'bm25':
                bm25_count += 1
            elif method == 'hybrid':
                hybrid_count += 1
            
            keywords = search_info.get('matched_keywords', [])
            all_keywords.extend(keywords)
        
        # Get top matched keywords
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_keywords = [kw for kw, _ in top_keywords]
        
        return {
            'semantic_count': semantic_count,
            'bm25_count': bm25_count,
            'hybrid_count': hybrid_count,
            'top_keywords': top_keywords,
            'total_keywords': len(set(all_keywords))
        } 